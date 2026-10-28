"""Launch one warmed-up SM90 sparse-decode call for Nsight Compute.

The call launches a split-KV main kernel and, when the scheduler produces
multiple splits, a combine kernel.  The accompanying Slurm script filters one
of those kernels by name, so this program deliberately performs no timing
loop after warm-up.
"""

import argparse

import torch

import flash_mla
import lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--s-q", type=int, default=2)
    parser.add_argument("--s-kv", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--h-q", type=int, choices=(64, 128), default=128)
    parser.add_argument("--d-qk", type=int, choices=(512, 576), default=576)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.s_q <= 0 or args.s_kv <= 0:
        raise ValueError("--batch-size, --s-q, and --s-kv must be positive")
    if args.topk <= 0 or args.topk > args.s_kv:
        raise ValueError("--topk must be positive and no larger than --s-kv")
    if args.topk % 64:
        raise ValueError("--topk must be a multiple of 64 for the SM90 decode kernel")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    raw_params = lib.RawTestParamForDecode(
        b=args.batch_size,
        h_q=args.h_q,
        s_q=args.s_q,
        h_kv=1,
        s_kv=args.s_kv,
        is_varlen=False,
        topk=args.topk,
        enable_attn_sink=False,
        block_size=args.block_size,
        d_qk=args.d_qk,
        d_v=512,
        check_correctness=False,
        num_runs=0,
        seed=20260831,
    )
    params = raw_params.to_test_param()
    testcase = lib.generate_testcase_for_decode(params)
    sched_meta, _ = flash_mla.get_mla_metadata()

    def run_decode():
        return lib.run_flash_mla_decode(params, testcase, sched_meta, None)

    print(
        "Profile case: "
        f"b={args.batch_size}, s_q={args.s_q}, s_kv={args.s_kv}, topk={args.topk}, "
        f"h_q={args.h_q}, d_qk={args.d_qk}, block_size={args.block_size}"
    )
    for _ in range(args.warmup):
        run_decode()
    torch.cuda.synchronize()

    # NCU captures the split-KV or combine launch from this final call.
    result = run_decode()
    torch.cuda.synchronize()

    stats = lib.count_flop_and_mem_vol_for_decode(params, testcase)
    total_splits = int(sched_meta.num_splits[-1].item()) if sched_meta.num_splits is not None else -1
    print(f"Scheduler splits: {total_splits} for {args.batch_size} requests")
    print(
        "Algorithmic model: "
        f"{stats.flop / 1e9:.3f} GFLOP, {stats.mem_vol / 1e6:.3f} MB "
        f"({stats.flop / stats.mem_vol:.1f} FLOP/B)"
    )
    print("Profiled sparse-decode call completed:", [tuple(x.shape) for x in result])


if __name__ == "__main__":
    main()
