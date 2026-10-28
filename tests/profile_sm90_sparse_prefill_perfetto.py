"""Run one SM90 sparse-prefill kernel and emit a Perfetto trace of its pipeline.

Requires the extension to be built with FLASH_MLA_TRACE_PERFETTO=1.

The device records one timestamped milestone per pipeline stage, plus explicit
BEGIN/END markers around each blocking wait (Q/KV load, cross-warpgroup sM/sS
sync). By default (--mode slices), each warpgroup is rendered as contiguous
``active`` and ``stall: ...`` bars; --mode instant renders every milestone as
a point. Tracing one CTA (--trace-only-cta 0) avoids trace instrumentation
between CTAs and keeps the Perfetto UI readable.

Example:
    FLASH_MLA_TRACE_PERFETTO=1 pip install -e .
    python tests/profile_sm90_sparse_prefill_perfetto.py \
        --s-q 128 --s-kv 65536 --topk 2048 --trace-only-cta 0 \
        --out trace.perfetto-trace

Then open trace.perfetto-trace at https://ui.perfetto.dev.
"""

import argparse
import os
import sys

import numpy as np
import torch

import lib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import emit_perfetto_trace  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-q", type=int, default=128)
    parser.add_argument("--s-kv", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--h-q", type=int, choices=(64, 128), default=128)
    parser.add_argument("--d-qk", type=int, choices=(512, 576), default=576)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--trace-only-cta", type=int, default=0,
        help="blockIdx.x to trace (one CTA is required for atomic-free tracing).",
    )
    parser.add_argument(
        "--out", default="trace.perfetto-trace",
        help="Output Perfetto trace path.",
    )
    parser.add_argument(
        "--mode", choices=("slices", "instant"), default="slices",
        help="slices (default) = labelled active/stall bars; "
        "instant = all milestones as point markers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk <= 0 or args.topk % 128:
        raise ValueError("--topk must be a positive multiple of 128")

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
    testcase = lib.generate_testcase(params)

    for _ in range(args.warmup):
        lib.run_flash_mla_sparse_fwd(params, testcase, return_p_sum=False)
    torch.cuda.synchronize()

    result = lib.run_flash_mla_sparse_fwd(
        params, testcase, return_p_sum=False,
        profile_trace=True, trace_only_cta=args.trace_only_cta,
    )
    torch.cuda.synchronize()

    out, max_logits, lse, trace_buffer, trace_num_events = result
    counts = trace_num_events.cpu().tolist()
    capacity = trace_buffer.numel() // (3 * emit_perfetto_trace.EVENT_DTYPE.itemsize)
    raw = trace_buffer.cpu().numpy().view(emit_perfetto_trace.EVENT_DTYPE)
    # The three regions are producer, WG0, WG1.  They are intentionally not
    # append-ordered: fixed regions avoid a global atomic in the kernel.
    chunks = [raw[role * capacity: role * capacity + min(int(n), capacity)]
              for role, n in enumerate(counts)]
    records = np.concatenate(chunks) if chunks else raw[:0]
    num_events = sum(int(n) for n in counts)
    n = len(records)
    dropped = sum(max(0, int(n) - capacity) for n in counts)

    blob = emit_perfetto_trace.build_perfetto_trace(records, mode=args.mode)
    with open(args.out, "wb") as f:
        f.write(blob)

    print(f"case: s_q={args.s_q} s_kv={args.s_kv} topk={args.topk} "
          f"h_q={args.h_q} d_qk={args.d_qk} trace_only_cta={args.trace_only_cta}")
    print(f"events recorded={num_events} kept={n} dropped={dropped} "
          f"(capacity/warpgroup={capacity}, counts={counts})")
    if n:
        t0 = int(records["ts"].min())
        t1 = int(records["ts"].max())
        print(f"timeline span: {(t1 - t0) / 1e6:.3f} ms "
              f"({(t1 - t0) / 1e9:.6f} s)")
        roles = {0: "producer", 1: "wg0", 2: "wg1"}
        for r in sorted(set(records["role"].tolist())):
            cnt = int((records["role"] == r).sum())
            print(f"  role {roles.get(int(r), r)}: {cnt} events")
    print(f"wrote {len(blob)} bytes to {args.out}")
    print("open it at https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
