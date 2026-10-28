"""Launch exactly one SM90 sparse-prefill kernel for Nsight Compute.

This program intentionally does no timing loop.  The Slurm job warms up the
extension and Nsight Compute profiles its single matching kernel launch.

Example (normally invoked by the accompanying .sbatch file):
    python tests/profile_sm90_sparse_prefill.py --s-q 4096 --s-kv 32768 --topk 2048
"""

import argparse
import dataclasses
from pathlib import Path
import random
import secrets

import torch

import lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-q", type=int, default=4096)
    parser.add_argument("--s-kv", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--h-q", type=int, choices=(64, 128), default=128)
    parser.add_argument("--d-qk", type=int, choices=(512, 576), default=576)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--regenerate-input-per-launch", action="store_true",
        help="Generate a distinct Q/KV/indices testcase before every sparse-prefill launch.",
    )
    parser.add_argument(
        "--input-seed", type=int,
        help="Base seed for input generation. With --regenerate-input-per-launch, "
             "the launch number is added; otherwise the default is the historical 20260827.",
    )
    parser.add_argument(
        "--event-iters", type=int, default=0,
        help="Measure this many launches with CUDA events instead of doing one profile launch.",
    )
    parser.add_argument(
        "--ready-wait-cycles", action="store_true",
        help="Return per-CTA clock64() totals for the four SM90 producer-ready waits.",
    )
    parser.add_argument(
        "--consumer-barrier-wait-cycles", action="store_true",
        help="Measure max clock64() waits for the four SM90 WG0/WG1 named barriers.",
    )
    parser.add_argument(
        "--consumer-barrier-wait-output", type=Path,
        help="Optional .pt output path for the raw [CTA, pair, barrier, warp] clock64 trace.",
    )
    return parser.parse_args()


def expected_traffic_bytes(args: argparse.Namespace) -> dict[str, int]:
    """Print requested bytes, not a prediction of physical DRAM traffic.

    KV is gathered once by every 64-Q-head CTA. L2 may serve repeated KV
    requests, so Nsight's DRAM bytes can be substantially smaller than the
    request volume below.  The output/LSE bytes include the main stores only.
    """
    bf16 = 2
    fp32 = 4
    ctas_per_q = args.h_q // 64
    kv_gather = args.s_q * ctas_per_q * args.topk * args.d_qk * bf16
    indices = args.s_q * ctas_per_q * args.topk * 4
    q_tma = args.s_q * args.h_q * args.d_qk * bf16
    out_tma = args.s_q * args.h_q * 512 * bf16
    stats = args.s_q * args.h_q * 2 * fp32  # max_logits and lse
    return {
        "producer indices": indices,
        "producer KV gather": kv_gather,
        "Q TMA read": q_tma,
        "epilogue O write": out_tma,
        "epilogue max/lse write": stats,
    }


def print_expected_traffic(args: argparse.Namespace) -> dict[str, int]:
    traffic = expected_traffic_bytes(args)
    print("Expected request-volume model (GiB; cache misses are not assumed):")
    for name, byte_count in traffic.items():
        print(f"  {name}={byte_count / 2**30:.3f}")
    print(f"  total requested={sum(traffic.values()) / 2**30:.3f}")
    print("  Note: QK / softmax / PV are fused and overlap with gather; their separate"
          " bandwidth cannot be obtained by dividing total kernel time into additive phases.")
    return traffic


def main() -> None:
    args = parse_args()
    if args.consumer_barrier_wait_output and not args.consumer_barrier_wait_cycles:
        raise ValueError("--consumer-barrier-wait-output requires --consumer-barrier-wait-cycles")
    if args.topk <= 0 or args.topk % 128:
        raise ValueError("--topk must be a positive multiple of 128 for the SM90 kernel")
    if args.input_seed is not None and not 0 <= args.input_seed < 2**63:
        raise ValueError("--input-seed must be in [0, 2**63)")

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    params = lib.TestParam(
        s_q=args.s_q,
        s_kv=args.s_kv,
        topk=args.topk,
        h_q=args.h_q,
        h_kv=1,
        d_qk=args.d_qk,
        d_v=512,
        seed=20260827,
        check_correctness=False,
        num_runs=0,
        have_attn_sink=True,
    )
    # The legacy profile has a fixed input.  The fresh-input mode instead uses
    # a unique seed per attention launch; its default base comes from OS
    # entropy, while --input-seed makes the whole sequence reproducible.
    base_seed = args.input_seed
    if base_seed is None:
        base_seed = secrets.randbits(63) if args.regenerate_input_per_launch else 20260827

    def make_testcase(launch_number: int) -> lib.Testcase:
        seed = base_seed + launch_number
        random.seed(seed)  # generate_testcase also uses Python's RNG for offsets.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return lib.generate_testcase(dataclasses.replace(params, seed=seed))

    testcase = make_testcase(0)
    print(f"Profile case: s_q={args.s_q}, s_kv={args.s_kv}, topk={args.topk}, "
          f"h_q={args.h_q}, d_qk={args.d_qk}")
    if args.regenerate_input_per_launch:
        print(f"Input policy: fresh testcase per launch (base seed={base_seed})")
    else:
        print(f"Input policy: one testcase reused for all launches (seed={base_seed})")
    traffic = print_expected_traffic(args)

    # These launches are intentionally skipped by NCU with --launch-skip.
    for launch_number in range(args.warmup):
        current_testcase = make_testcase(launch_number) if args.regenerate_input_per_launch else testcase
        lib.run_flash_mla_sparse_fwd(params, current_testcase, return_p_sum=False)
    torch.cuda.synchronize()

    if args.event_iters:
        if args.event_iters < 1:
            raise ValueError("--event-iters must be positive")
        if args.regenerate_input_per_launch:
            # Synchronize after generation and time each attention launch on
            # its own, so randn/index construction is not charged to the kernel.
            elapsed_ms = 0.0
            for iteration in range(args.event_iters):
                current_testcase = make_testcase(args.warmup + iteration)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = lib.run_flash_mla_sparse_fwd(params, current_testcase, return_p_sum=False)
                end.record()
                end.synchronize()
                elapsed_ms += start.elapsed_time(end)
            elapsed_ms /= args.event_iters
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.event_iters):
                result = lib.run_flash_mla_sparse_fwd(params, testcase, return_p_sum=False)
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / args.event_iters
        elapsed_s = elapsed_ms / 1e3
        print(f"CUDA-event mean kernel time: {elapsed_ms:.3f} ms ({args.event_iters} launches)")
        print("Effective request bandwidth (stage bytes / whole fused-kernel time):")
        for name, byte_count in traffic.items():
            print(f"  {name}: {byte_count / elapsed_s / 1e9:.2f} GB/s")
        print(f"  total requested: {sum(traffic.values()) / elapsed_s / 1e9:.2f} GB/s")
        print("  These rates overlap in time; do not sum them or interpret them as"
              " independent DRAM/L2 bandwidth measurements.")
    else:
        # Keep the final result live until the launch has completed. This is the
        # one Nsight Compute collects after the warmup launches.
        consumer_barrier_wait_cycles = None
        if args.consumer_barrier_wait_cycles:
            num_ctas = args.s_q * (args.h_q // 64)
            num_pairs = args.topk // 128
            consumer_barrier_wait_cycles = torch.zeros(
                (num_ctas, num_pairs, 4, 4), device="cuda", dtype=torch.int64
            )
        current_testcase = make_testcase(args.warmup) if args.regenerate_input_per_launch else testcase
        result = lib.run_flash_mla_sparse_fwd(
            params, current_testcase, return_p_sum=False,
            consumer_barrier_wait_cycles=consumer_barrier_wait_cycles,
        )
        torch.cuda.synchronize()
        print("Profiled sparse-prefill launch completed:", [tuple(x.shape) for x in result])
        if args.ready_wait_cycles:
            ready_wait_cycles = result[3].cpu().to(torch.float64)
            waits_per_cta = args.topk // 128
            labels = ("K0-left", "K0-right", "K1-right", "K1-left")
            print(f"Ready-wait clock64 totals per CTA ({waits_per_cta} waits per column):")
            for column, label in enumerate(labels):
                per_wait = ready_wait_cycles[:, column] / waits_per_cta
                print(
                    f"  {label}: mean={per_wait.mean().item():.1f} cycles/wait, "
                    f"p50={per_wait.quantile(0.50).item():.1f}, "
                    f"p95={per_wait.quantile(0.95).item():.1f}, "
                    f"max={per_wait.max().item():.1f}"
                )
        if args.consumer_barrier_wait_cycles:
            assert consumer_barrier_wait_cycles is not None
            trace_i64 = consumer_barrier_wait_cycles.cpu()
            trace = trace_i64.to(torch.float64)
            labels = ("B0 WG1 waits P0", "B1 WG0 waits P1", "B2 WG1 waits S0", "B3 WG0 waits S1")
            print("Consumer named-barrier max clock64 waits (per CTA and block pair):")
            for column, label in enumerate(labels):
                values = trace[:, :, column, :].amax(dim=-1).flatten()
                print(
                    f"  {label}: mean={values.mean().item():.1f} cycles, "
                    f"p50={values.quantile(0.50).item():.1f}, "
                    f"p95={values.quantile(0.95).item():.1f}, "
                    f"max={values.max().item():.1f}"
                )
            if args.consumer_barrier_wait_output:
                args.consumer_barrier_wait_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(trace_i64, args.consumer_barrier_wait_output)
                print(f"Saved raw consumer-barrier trace: {args.consumer_barrier_wait_output}")


if __name__ == "__main__":
    main()
