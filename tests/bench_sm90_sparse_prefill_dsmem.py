"""Slurm-only baseline/fine-grained DSMEM benchmark for SM90 BF16 sparse prefill."""

import argparse
from typing import Callable, Tuple

import torch

import flash_mla
import kernelkit as kk
import lib

D_QK = 576
D_V = 512
H_Q = 128

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-q", type=int, default=128, help="128 models B=128, decode s_q=1")
    parser.add_argument("--s-kv", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.set_defaults(check=True)
    return parser.parse_args()

def event_time_ms(fn: Callable[[], Tuple[torch.Tensor, ...]], iters: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters

def main() -> None:
    args = parse_args()
    if args.topk <= 0 or args.topk % 128 or args.topk > args.s_kv:
        raise ValueError("topk must be a positive multiple of 128 and no larger than s_kv")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    p = lib.TestParam(
        s_q=args.s_q,
        s_kv=args.s_kv,
        topk=args.topk,
        h_q=H_Q,
        h_kv=1,
        d_qk=D_QK,
        d_v=D_V,
        seed=args.seed,
        check_correctness=False,
        num_runs=0,
        have_attn_sink=False,
    )
    t = lib.generate_testcase(p)

    def direct() -> Tuple[torch.Tensor, ...]:
        return flash_mla.flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale)

    def dsmem_pipe() -> Tuple[torch.Tensor, ...]:
        return flash_mla.flash_mla_sparse_fwd_dsmem_crossover_pipelined(t.q, t.kv, t.indices, t.sm_scale)
    kv_bytes = args.s_q * args.topk * D_QK * 2
    print(f"case: s_q={args.s_q} s_kv={args.s_kv} topk={args.topk}")
    print(f"direct global-KV requests:    {2 * kv_bytes / 2**20:.3f} MiB")
    print(f"crossover global-KV requests: {kv_bytes / 2**20:.3f} MiB")
    print(f"crossover DSMEM exchange:     {kv_bytes / 2**20:.3f} MiB")

    for _ in range(args.warmup):
        direct()
        dsmem_pipe()
    torch.cuda.synchronize()

    direct_out, dsmem_pipe_out = direct(), dsmem_pipe()
    torch.cuda.synchronize()
    if args.check:
        ok = kk.check_is_allclose("dsmem_pipe.out", dsmem_pipe_out[0], direct_out[0], abs_tol=1e-3, rel_tol=3.01 / 128, cos_diff_tol=7e-6)
        ok &= kk.check_is_allclose("dsmem_pipe.max_logits", dsmem_pipe_out[1], direct_out[1], abs_tol=1e-6, rel_tol=2.01 / 65536)
        ok &= kk.check_is_allclose("dsmem_pipe.lse", dsmem_pipe_out[2], direct_out[2], abs_tol=1e-6, rel_tol=2.01 / 65536)
        if not ok:
            raise SystemExit("dsmem_pipe correctness check failed")

    direct_ms = event_time_ms(direct, args.iters)
    dsmem_pipe_ms = event_time_ms(dsmem_pipe, args.iters)
    print(f"direct:     {direct_ms * 1e3:.1f} us")
    print(f"dsmem_pipe: {dsmem_pipe_ms * 1e3:.1f} us ({direct_ms / dsmem_pipe_ms:.3f}x vs direct)")


if __name__ == "__main__":
    main()
