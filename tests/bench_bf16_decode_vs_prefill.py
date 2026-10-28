"""
Compare the two ways of running a *sparse decode* step with a bfloat16 KV cache on SM90:

  A) `flash_mla_sparse_fwd`  -- the sparse **prefill** kernel (`sparse_attn_fwd_kernel`),
     fed the decode workload by flattening (b, s_q) into its s_q dimension and handing it a
     flat view of the paged KV cache.
  B) `flash_mla_with_kvcache(..., is_fp8_kvcache=False)` -- the bf16 path of the sparse
     **decode** kernel (`flash_fwd_splitkv_mla_fp8_sparse_kernel` with IS_BF16_KV == true),
     followed by `flash_fwd_mla_combine_kernel`.

Both paths read the same KV bytes and do the same FLOPs; they differ only in how the work is
parallelised:

  * prefill launches `(h_q/64) * b * s_q` CTAs and has **no split-KV**, so with a small batch
    it cannot fill the GPU no matter how large `topk` is;
  * decode splits the topk dimension across SMs (`DecodingSchedMeta`) and runs a 2-CTA cluster
    with a DSMEM crossover when h_q == 128.

So the interesting axis is batch size, which is what the default sweep varies.

Usage (see also ~/slurm/flashmla_bf16_vs_prefill.slurm):

    python tests/bench_bf16_decode_vs_prefill.py
    python tests/bench_bf16_decode_vs_prefill.py --bsz 1 2 4 --h-q 128 --topk 2048
    python tests/bench_bf16_decode_vs_prefill.py --with-fp8      # add the fp8 decode baseline
"""

import argparse
import dataclasses
from typing import List, Optional

import rich.console
import rich.table
import torch

import kernelkit as kk
import flash_mla

import lib
import ref

DECODE_SPLITKV_KERNEL = "flash_fwd_splitkv_mla_fp8_sparse_kernel"
DECODE_COMBINE_KERNEL = "flash_fwd_mla_combine_kernel"
PREFILL_KERNEL = "sparse_attn_fwd"

D_QK = 576
D_V = 512
H_KV = 1


@dataclasses.dataclass
class Config:
    b: int
    h_q: int = 128
    s_q: int = 2
    s_kv: int = 32768
    topk: int = 2048
    block_size: int = 64
    num_runs: int = 10
    # Checking against the PyTorch reference materialises a [b, s_q, topk, d_qk] fp32 tensor,
    # which gets expensive fast. A/B cross-checking the two kernels is always cheap, so the
    # reference is only used on the small-batch configs.
    check_vs_ref: bool = False

    def __post_init__(self):
        # The prefill kernel processes 2*B_TOPK == 128 topk entries per iteration and 64 heads
        # per CTA (csrc/sm90/prefill/sparse/phase1.cuh:576-579).
        assert self.topk % 128 == 0, f"topk must be a multiple of 128 for the prefill kernel, got {self.topk}"
        assert self.h_q % 64 == 0, f"h_q must be a multiple of 64, got {self.h_q}"
        assert self.topk <= self.s_kv, "topk > s_kv would leave invalid indices, which skews the byte count"


@dataclasses.dataclass
class Measurement:
    time_us: float
    tflops: float
    gBps: float
    # Per-kernel breakdown, in us. For the decode path this separates the splitkv kernel from
    # the combine kernel, which is what tells you whether split-KV is costing anything.
    parts: dict = dataclasses.field(default_factory=dict)


def _to_raw_param(cfg: Config, is_fp8_kvcache: bool, seed: int) -> lib.RawTestParamForDecode:
    return lib.RawTestParamForDecode(
        b=cfg.b,
        h_q=cfg.h_q,
        s_q=cfg.s_q,
        h_kv=H_KV,
        s_kv=cfg.s_kv,
        # Fixed-length so every one of the `topk` indices is valid: a varlen batch would make
        # the number of KV bytes read data-dependent and the A/B numbers harder to read.
        is_varlen=False,
        topk=cfg.topk,
        enable_attn_sink=False,
        block_size=cfg.block_size,
        d_qk=D_QK,
        d_v=D_V,
        is_fp8_kvcache=is_fp8_kvcache,
        check_correctness=True,
        num_runs=cfg.num_runs,
        seed=seed,
    )


def _contiguous_view(name: str, t: torch.Tensor) -> torch.Tensor:
    """
    `lib` deliberately hands out non-contiguous tensors to exercise the stride handling. For a
    benchmark we want both paths to see exactly the same, contiguous, serving-shaped inputs, and
    the prefill path additionally needs to `view()` (not `reshape()`) the decode shapes.
    """
    if t.is_contiguous():
        return t
    print(f"  (materialising a contiguous copy of {name}, shape {tuple(t.shape)})")
    return t.contiguous()


def _measure(fn, num_runs: int, kernel_names: List[str], flop: float, mem_vol: float) -> Optional[Measurement]:
    if num_runs == 0:
        return None
    result = kk.bench_kineto(fn, num_tests=num_runs)
    available = result.get_kernel_names()
    present = [n for n in kernel_names if any(n in s for s in available)]
    if not present:
        raise RuntimeError(
            f"None of {kernel_names} showed up in the profile. Kernels seen: {available}"
        )
    if kk.is_using_profiling_tools():
        time_s = 1.0
    elif len(present) == 1:
        time_s = result.get_kernel_time(present[0])
    else:
        # More than one kernel in the sequence (decode: splitkv + combine): measure the span from
        # the first kernel's start to the last kernel's end so the gap between them is counted.
        time_s = result.get_e2e_time(present[0], present[-1])
    parts = {n: result.get_kernel_time(n) * 1e6 for n in present} if not kk.is_using_profiling_tools() else {}
    return Measurement(time_s * 1e6, flop / time_s / 1e12, mem_vol / time_s / 1e9, parts)


@torch.inference_mode()
def run_one(cfg: Config, seed: int, with_fp8: bool) -> dict:
    print("================")
    print(
        f"b={cfg.b} h_q={cfg.h_q} s_q={cfg.s_q} s_kv={cfg.s_kv} topk={cfg.topk} "
        f"block_size={cfg.block_size} d_qk={D_QK} (bf16 KV cache)"
    )
    torch.cuda.empty_cache()

    p = _to_raw_param(cfg, is_fp8_kvcache=False, seed=seed).to_test_param()
    t = lib.generate_testcase_for_decode(p)

    b, s_q, h_q, topk = cfg.b, cfg.s_q, cfg.h_q, cfg.topk

    # ---- Inputs shared by both paths -------------------------------------------------------
    q_decode = _contiguous_view("q", t.q)                                   # [b, s_q, h_q, d_qk]
    kv_decode = _contiguous_view("kv", t.kv_scope.get_kvcache_for_flash_mla())
    idx_decode = _contiguous_view("indices", t.kv_scope.indices_in_kvcache)  # [b, s_q, topk]
    assert kv_decode.dtype == torch.bfloat16

    # The prefill kernel has no batch dimension, so (b, s_q) collapses into its s_q dim, and the
    # paged cache becomes one flat [total_tokens, h_kv, d_qk] sequence. That works because
    # `indices_in_kvcache` is already a *flat* index into the blocked cache (block_idx *
    # block_size + offset), which is exactly what prefill wants -- see tests/ref.py:70.
    q_prefill = q_decode.view(b * s_q, h_q, D_QK)
    kv_prefill = kv_decode.view(-1, H_KV, D_QK)
    idx_prefill = idx_decode.view(b * s_q, H_KV, topk)

    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    prefill_ctas = (h_q // 64) * b * s_q
    print(
        f"  prefill grid: {prefill_ctas} CTAs on {num_sms} SMs "
        f"({prefill_ctas / num_sms:.2f} waves, no split-KV)"
    )

    sched_meta, _ = flash_mla.get_mla_metadata()

    def run_decode():
        return flash_mla.flash_mla_with_kvcache(
            q_decode, kv_decode, None, None, D_V,
            sched_meta, None,
            t.sm_scale, False, False,   # causal=False, is_fp8_kvcache=False
            idx_decode,
        )

    def run_prefill():
        return flash_mla.flash_mla_sparse_fwd(
            q_prefill, kv_prefill, idx_prefill, t.sm_scale, D_V
        )

    # Run both once up front so the correctness answers are produced before the perf loops
    # allocate over them (same reasoning as test_flash_mla_sparse_decoding.py).
    torch.cuda.synchronize()
    decode_out, decode_lse = run_decode()
    prefill_out, _prefill_max_logits, prefill_lse = run_prefill()
    torch.cuda.synchronize()

    # ---- Performance -----------------------------------------------------------------------
    stats = lib.count_flop_and_mem_vol_for_decode(p, t)
    flop, mem_vol = stats.flop, stats.mem_vol
    print(f"  Compute/Memory: {flop / mem_vol:.1f} FLOP/B  ({mem_vol / 1e9:.2f} GB per call)")

    decode_perf = _measure(
        run_decode, cfg.num_runs,
        [DECODE_SPLITKV_KERNEL, DECODE_COMBINE_KERNEL], flop, mem_vol,
    )
    prefill_perf = _measure(run_prefill, cfg.num_runs, [PREFILL_KERNEL], flop, mem_vol)

    fp8_perf = None
    if with_fp8:
        fp8_perf = _measure_fp8_decode(cfg, seed)

    for name, m in (("decode(bf16)", decode_perf), ("prefill(bf16)", prefill_perf), ("decode(fp8)", fp8_perf)):
        if m is not None:
            breakdown = "  ".join(f"{k.split('_kernel')[0].split('flash_fwd_')[-1]}={v:.1f}" for k, v in m.parts.items())
            print(f"  {name:<14} {m.time_us:8.1f} us   {m.tflops:6.1f} TFLOPS   {m.gBps:6.0f} GB/s   [{breakdown}]")
    if decode_perf is not None and prefill_perf is not None:
        print(f"  prefill / decode: {prefill_perf.time_us / decode_perf.time_us:.2f}x")

    # How many requests the scheduler split. `num_splits` is a [b+1] prefix sum, so num_splits[-1]
    # is the total split count; anything above `b` means that many extra splits went through the
    # fp32 o_accum + combine round-trip instead of the direct bf16 TMA store.
    total_splits = int(sched_meta.num_splits[-1].item()) if sched_meta.num_splits is not None else -1
    print(f"  split-KV: {total_splits} splits for {b} requests "
          f"({total_splits - b} extra -> o_accum + combine)")

    # ---- Correctness -----------------------------------------------------------------------
    # A vs B. Both consume an unquantised bf16 cache, so the only difference is the accumulation
    # order; the tolerances are the ones the decode test already uses.
    is_correct = True
    is_correct &= kk.check_is_allclose(
        "out (prefill vs decode)",
        prefill_out.view(b, s_q, h_q, D_V), decode_out,
        abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6,
    )
    # decode returns lse as [b, h_q, s_q]; prefill as [b*s_q, h_q]. prefill reports -inf for a
    # q token with nothing to attend to, decode reports +inf (see test_flash_mla_sparse_prefill.py:23).
    prefill_lse_norm = prefill_lse.clone()
    prefill_lse_norm[prefill_lse_norm == float("-inf")] = float("+inf")
    is_correct &= kk.check_is_allclose(
        "lse (prefill vs decode)",
        prefill_lse_norm, decode_lse.transpose(1, 2).reshape(b * s_q, h_q),
        abs_tol=1e-6, rel_tol=8.01 / 65536,
    )

    if cfg.check_vs_ref:
        ref_out, ref_lse = ref.ref_sparse_attn_decode(p, t)
        is_correct &= kk.check_is_allclose(
            "out (decode vs ref)", decode_out, ref_out,
            abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6,
        )
        is_correct &= kk.check_is_allclose(
            "lse (decode vs ref)", decode_lse, ref_lse,
            abs_tol=1e-6, rel_tol=8.01 / 65536,
        )
        is_correct &= kk.check_is_allclose(
            "out (prefill vs ref)", prefill_out.view(b, s_q, h_q, D_V), ref_out,
            abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6,
        )

    print(f"  correctness: {'PASS' if is_correct else 'FAIL'}")

    return {
        "cfg": cfg,
        "compute_memory_ratio": flop / mem_vol,
        "prefill_ctas": prefill_ctas,
        "total_splits": total_splits,
        "decode": decode_perf,
        "prefill": prefill_perf,
        "fp8": fp8_perf,
        "is_correct": is_correct,
    }


@torch.inference_mode()
def _measure_fp8_decode(cfg: Config, seed: int) -> Optional[Measurement]:
    """
    The fp8 decode path, as the baseline the bf16 path is a 1.76x-KV-bytes trade against. It
    needs its own testcase because the KV cache layout differs, so it is not part of the A/B
    correctness comparison.
    """
    p = _to_raw_param(cfg, is_fp8_kvcache=True, seed=seed).to_test_param()
    t = lib.generate_testcase_for_decode(p)
    q = _contiguous_view("q", t.q)
    kv = t.kv_scope.get_kvcache_for_flash_mla()
    idx = _contiguous_view("indices", t.kv_scope.indices_in_kvcache)
    sched_meta, _ = flash_mla.get_mla_metadata()

    def run():
        return flash_mla.flash_mla_with_kvcache(
            q, kv, None, None, D_V, sched_meta, None,
            t.sm_scale, False, True, idx,
        )

    torch.cuda.synchronize()
    run()
    torch.cuda.synchronize()
    stats = lib.count_flop_and_mem_vol_for_decode(p, t)
    m = _measure(
        run, cfg.num_runs, [DECODE_SPLITKV_KERNEL, DECODE_COMBINE_KERNEL],
        stats.flop, stats.mem_vol,
    )
    del t
    torch.cuda.empty_cache()
    return m


def gen_configs(args) -> List[Config]:
    configs = []
    for h_q in args.h_q:
        for topk in args.topk:
            for b in args.bsz:
                configs.append(Config(
                    b=b, h_q=h_q, s_q=args.s_q, s_kv=args.s_kv, topk=topk,
                    block_size=args.block_size, num_runs=args.runs,
                    check_vs_ref=args.check_ref and b <= args.check_ref_max_bsz,
                ))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bsz", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128],
                        help="batch sizes to sweep (this is the axis that matters: prefill has no split-KV)")
    parser.add_argument("--h-q", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--topk", type=int, nargs="+", default=[2048],
                        help="must be a multiple of 128 (the prefill kernel's 2*B_TOPK)")
    parser.add_argument("--s-q", type=int, default=2)
    parser.add_argument("--s-kv", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--with-fp8", action="store_true",
                        help="also time the fp8 decode path as a baseline")
    parser.add_argument("--check-ref", action="store_true", default=True,
                        help="also validate against the PyTorch reference on small batches")
    parser.add_argument("--no-check-ref", dest="check_ref", action="store_false")
    parser.add_argument("--check-ref-max-bsz", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(32)

    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}, {props.multi_processor_count} SMs, "
          f"{props.total_memory / 1e9:.0f} GB")
    if props.major != 9:
        print(f"{kk.colors['RED_BG']}WARNING: the bf16 sparse decode path is SM90-only "
              f"(this device is sm{props.major}{props.minor}){kk.colors['CLEAR']}")

    configs = gen_configs(args)
    print(f"{kk.colors['CYAN_BG']}{len(configs)} configs to run{kk.colors['CLEAR']}")

    results = []
    for i, cfg in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] ", end="")
        results.append(run_one(cfg, seed=1000 + i, with_fp8=args.with_fp8))

    console = rich.console.Console(width=140)
    table = rich.table.Table(show_header=True, header_style="bold cyan")
    for col in ["h_q", "Bsz", "sq", "sk", "topk", "C/M", "CTAs", "splits",
                "decode us", "splitkv", "combine", "decode GB/s", "prefill us", "prefill GB/s"]:
        table.add_column(col)
    if args.with_fp8:
        table.add_column("fp8 us")
    table.add_column("pre/dec")
    table.add_column(" ")

    for r in results:
        cfg: Config = r["cfg"]
        dec, pre, fp8 = r["decode"], r["prefill"], r["fp8"]
        splitkv = dec.parts.get(DECODE_SPLITKV_KERNEL) if dec else None
        combine = dec.parts.get(DECODE_COMBINE_KERNEL) if dec else None
        row = [
            str(cfg.h_q), str(cfg.b), str(cfg.s_q), str(cfg.s_kv), str(cfg.topk),
            f"{r['compute_memory_ratio']:3.0f}", str(r["prefill_ctas"]), str(r["total_splits"]),
            f"{dec.time_us:7.1f}" if dec else "-",
            f"{splitkv:7.1f}" if splitkv else "-",
            f"{combine:6.1f}" if combine else "-",
            f"{dec.gBps:5.0f}" if dec else "-",
            f"{pre.time_us:7.1f}" if pre else "-",
            f"{pre.gBps:5.0f}" if pre else "-",
        ]
        if args.with_fp8:
            row.append(f"{fp8.time_us:7.1f}" if fp8 else "-")
        row.append(f"{pre.time_us / dec.time_us:.2f}x" if (dec and pre) else "-")
        row.append("" if r["is_correct"] else "X")
        table.add_row(*row)
    console.print(table)

    num_correct = sum(1 for r in results if r["is_correct"])
    if num_correct == len(results):
        print(f"{kk.colors['GREEN_BG']}{num_correct}/{len(results)} configs correct{kk.colors['CLEAR']}")
    else:
        print(f"{kk.colors['RED_BG']}{num_correct}/{len(results)} configs correct{kk.colors['CLEAR']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
