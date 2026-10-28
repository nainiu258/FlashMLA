"""CuTeDSL reference port of SM90 sparse-prefill phase 1.

This is deliberately a *separate*, opt-in implementation of
``phase1.cuh``.  It does not alter the pre-compiled FlashMLA extension or
its dispatch path.  **It is currently a scalar semantic reference, not the
three-warp-group performance port.**  In particular,
``indirect_cp_async_cg_16b`` below is a DSL wrapper which is not yet called
by ``kernel``.  The initial port prioritizes an easily auditable CuTeDSL
implementation of the same sparse-attention semantics over matching the
hand-scheduled C++ kernel's performance:

* one CuTeDSL thread owns one (query-token, query-head) output row;
* it gathers the selected KV rows through ``indices``;
* it performs online softmax in scalar FP32 and writes the 512-value output.

The C++ kernel uses three warp groups, TMA for Q/O, GMMA and a double-buffered
``cp.async`` gather pipeline.  CuTeDSL does not package the *indirect*
``cp.async`` gather as a normal tensor-copy primitive, so the hybrid path
below wraps that one instruction with inline PTX.  Keeping this semantically
complete scalar port separate gives us a correctness oracle before replacing
it with a warp-specialized CuTeDSL pipeline.

Supported initial contract (matching the common SM90 V3.2 prefill case):

    q:           [s_q, h_q, 576], bfloat16
    kv:          [s_kv, h_kv, 576], bfloat16
    indices:     [s_q, h_kv, topk], int32
    attn_sink:   [h_q], float32

``kv[..., :512]`` is the value vector, as in ``phase1.cuh``.  A negative or
out-of-range gathered index is masked.  ``topk_length`` is intentionally not
part of this first path; pass a padded ``indices`` tensor with invalid entries
after the effective top-k instead.

Tensor-shape legend for the eventual 3-warp-group kernel (``B_H=64``,
``B_TOPK=64``):

    gQ           [s_q, h_q, 576]          bf16, global query tensor
    gKV          [s_kv, h_kv, 576]        bf16, global KV tensor
    gIndices     [s_q, h_kv, topk]        int32, sparse token ids
    gOut         [s_q, h_q, 512]          bf16
    gMax/gLse    [s_q, h_q]               fp32

    sQ           [64, 576]                bf16, TMA-loaded query-head tile
    sKV[stage]   [64, 576]                bf16, gathered KV tile; two stages
    rP           [64, 64]                 fp32, QK WGMMA accumulator
    rS           [64, 64]                 bf16, softmax probabilities for PV
    rO[WG0/1]    [64, 256]                fp32, PV WGMMA accumulator per WG
    CTA O tile   [64, 512]                bf16, two 256-wide WG output halves

The original C++ ownership is deliberately asymmetric:

    WG0/WG1: one elected thread launches the Q TMA into ``sQ``; both groups
             consume ``sQ`` and ``sKV`` with WGMMA.
    WG2:     producer only -- it does *not* load Q.  It reads
             ``gIndices[s_q, 0, :]`` and gathers ``gKV`` into ``sKV`` with
             ``cp.async``.

For every two 64-token top-k blocks, WG2 executes this four-part producer
schedule.  ``L`` means columns ``[0,256)`` and ``R`` means ``[256,576)``:

    1. V0L: sKV[0, :, L] <- indices[block + 0]  -> bar_k0_ready[0]
    2. V1R: sKV[1, :, R] <- indices[block + 1]  -> bar_k1_ready[1]
    3. V0R: sKV[0, :, R] <- indices[block + 0]  -> bar_k0_ready[1]
    4. V1L: sKV[1, :, L] <- indices[block + 1]  -> bar_k1_ready[0]

Each of the 128 producer threads belongs to one of 16 groups of 8.  A group
loads four token rows, and a thread loads one aligned 16-byte vector in each
64-wide tile.  This exact producer loop and its mbarrier hand-off are still
to be wired into the warp-specialized CuTeDSL kernel; the scalar ``kernel``
below intentionally has no ``wg == 2`` branch.  ``sQ`` and ``gOut`` are
affine and use TMA; the KV gather is indirect and therefore uses ``cp.async``.

Usage (after installing the repository's bundled CuTeDSL package):

    from csrc.sm90.prefill.sparse.phase1_cute_dsl import sparse_prefill
    out, max_logits, lse = sparse_prefill(q, kv, indices, sm_scale, attn_sink)

CuTeDSL compiles this Python module on first use.  It is therefore unsuitable
for the existing no-JIT benchmark path until a compiled-artifact integration
is added separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math as _math
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Boolean
from cutlass.cutlass_dsl import T, dsl_user_op


D_QK = 576
D_V = 512
LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class ThreeWarpGroupPlan:
    """Execution roles preserved from the C++ SM90 phase-1 kernel.

    A Hopper warp group has 128 threads.  The first two groups own the QK/PV
    WGMMA work over ``rP:[64,64]`` and one ``rO:[64,256]`` half each.  They
    also issue the single Q TMA.  The final group owns the ``sKV:[64,576]``
    indirect KV gather.  TMA is only applicable to the affine Q/O tensors.
    KV remains an indirect gather and therefore uses ``cp.async`` exactly as
    the C++ implementation does.
    """

    threads_per_warp_group: int = 128
    producer_warp_group: int = 2
    compute_warp_group_0: int = 0
    compute_warp_group_1: int = 1
    q_tma_issuer_warp_groups: tuple[int, int] = (0, 1)
    producer_threads_per_index_group: int = 8
    producer_index_groups: int = 16
    producer_rows_per_index_group: int = 4
    qk_mma_shape: tuple[int, int, int] = (64, 64, 16)
    pv_mma_shape: tuple[int, int, int] = (64, 256, 16)
    kv_tile_shape: tuple[int, int] = (64, D_QK)

    # Exact order used by phase1.cuh's WG2 producer for two top-k blocks.
    # (KV-buffer, top-k-block-relative-index, D_QK tile range, ready barrier)
    producer_copy_order: tuple[tuple[int, int, tuple[int, int], str], ...] = (
        (0, 0, (0, 4), "bar_k0_ready[0]"),
        (1, 1, (4, 9), "bar_k1_ready[1]"),
        (0, 0, (4, 9), "bar_k0_ready[1]"),
        (1, 1, (0, 4), "bar_k1_ready[0]"),
    )

    @property
    def threads_per_block(self) -> int:
        return 3 * self.threads_per_warp_group


THREE_WARP_GROUP_PLAN = ThreeWarpGroupPlan()


@dsl_user_op
def indirect_cp_async_cg_16b(
    dst_smem: cute.Pointer,
    src_gmem: cute.Pointer,
    pred: Boolean,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue the producer's predicated 16-byte indirect KV gather.

    CuTeDSL's ordinary TMA tensor maps require an affine source layout.  The
    source address here is ``kv + indices[...] * stride_kv`` and therefore
    cannot be represented by a TMA map.  This is the one intentional inline
    PTX escape hatch in the hybrid port.  Q and O remain TMA tensors, while
    WGMMA and all layout construction remain CuTeDSL operations.
    """
    dst_i32 = dst_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    src_i64 = src_gmem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    pred_i32 = Boolean(pred).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [dst_i32, src_i64, pred_i32],
        """{
            .reg .pred p;
            setp.ne.b32 p, $2, 0;
            @p   cp.async.cg.shared.global [$0], [$1], 16;
            @!p  cp.async.cg.shared.global [$0], [$1], 16, 0;
        }""",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class Sm90SparsePrefillCuTeDsl:
    """Semantic CuTeDSL implementation for the common SM90 sparse-prefill case.

    The launch shape is intentionally simple: a linear thread index selects a
    single (s_q, h_q) row.  This makes the indirect gather and the online
    softmax mapping exact and keeps the initial DSL port independent of the
    C++ extension's ``SparseAttnFwdParams`` ABI.
    """

    def __init__(self, threads_per_block: int = 128) -> None:
        if threads_per_block <= 0 or threads_per_block % 32:
            raise ValueError("threads_per_block must be a positive multiple of 32")
        self.threads_per_block = threads_per_block

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        kv: cute.Tensor,
        indices: cute.Tensor,
        attn_sink: cute.Tensor,
        out: cute.Tensor,
        max_logits: cute.Tensor,
        lse: cute.Tensor,
        sm_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(q.element_type != cutlass.BFloat16):
            raise TypeError("q must be bfloat16")
        if cutlass.const_expr(kv.element_type != cutlass.BFloat16):
            raise TypeError("kv must be bfloat16")
        if cutlass.const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be bfloat16")
        if cutlass.const_expr(indices.element_type != cutlass.Int32):
            raise TypeError("indices must be int32")
        if cutlass.const_expr(attn_sink.element_type != cutlass.Float32):
            raise TypeError("attn_sink must be float32")
        if cutlass.const_expr(max_logits.element_type != cutlass.Float32):
            raise TypeError("max_logits must be float32")
        if cutlass.const_expr(lse.element_type != cutlass.Float32):
            raise TypeError("lse must be float32")

        if cutlass.const_expr(q.shape[2] != D_QK or kv.shape[2] != D_QK):
            raise ValueError("the initial CuTeDSL port supports d_qk=576 only")
        if cutlass.const_expr(out.shape[2] != D_V):
            raise ValueError("the initial CuTeDSL port supports d_v=512 only")
        if cutlass.const_expr(q.shape[0] != indices.shape[0]):
            raise ValueError("q and indices must have the same s_q")
        if cutlass.const_expr(kv.shape[1] != indices.shape[1]):
            raise ValueError("kv and indices must have the same h_kv")
        if cutlass.const_expr(q.shape[0] != out.shape[0] or q.shape[1] != out.shape[1]):
            raise ValueError("out must have shape [s_q, h_q, 512]")

        rows = q.shape[0] * q.shape[1]
        self.kernel(q, kv, indices, attn_sink, out, max_logits, lse, sm_scale).launch(
            grid=(cute.ceil_div(rows, self.threads_per_block), 1, 1),
            block=(self.threads_per_block, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        kv: cute.Tensor,
        indices: cute.Tensor,
        attn_sink: cute.Tensor,
        out: cute.Tensor,
        max_logits: cute.Tensor,
        lse: cute.Tensor,
        sm_scale: cutlass.Float32,
    ):
        """Gather sparse KV rows, apply online softmax, and write one output row.

        Tensor views in this scalar reference:

        * ``q[q_idx, q_head_idx, :]``: ``[576]``
        * ``indices[q_idx, kv_head_idx, :]``: ``[topk]``
        * ``kv[token_idx, kv_head_idx, :]``: ``[576]``
        * ``out[q_idx, q_head_idx, :]``: ``[512]``
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        linear_row = bidx * self.threads_per_block + tidx
        row_count = q.shape[0] * q.shape[1]# seqlen num_head

        if linear_row >= row_count:
            return

        q_idx = linear_row // q.shape[1] 
        q_head_idx = linear_row - q_idx * q.shape[1]
        # GQA mapping.  The production SM90 sparse-prefill path normally has
        # h_kv == 1, but this keeps the reference valid for grouped heads.
        kv_head_idx = q_head_idx % kv.shape[1] 
        topk = indices.shape[2]
        sm_scale_log2 = sm_scale * cutlass.Float32(LOG2_E)

        # Pass 1: max(Q K^T).  The value of every invalid index is -inf.
        row_max = cutlass.Float32(-3.402823466e38)
        for token_slot in cutlass.range(topk):
            token_idx = indices[q_idx, kv_head_idx, token_slot]
            score = cutlass.Float32(-3.402823466e38)
            if token_idx >= 0 and token_idx < kv.shape[0]:
                score = cutlass.Float32(0.0)
                for d in cutlass.range(D_QK):
                    score += cutlass.Float32(q[q_idx, q_head_idx, d]) * cutlass.Float32(
                        kv[token_idx, kv_head_idx, d]
                    )
            row_max = cute.arch.fmax(row_max, score)

        # Match phase1.cuh's all-invalid behavior: O=0, max=-inf, lse=+inf.
        if row_max == cutlass.Float32(-3.402823466e38):
            max_logits[q_idx, q_head_idx] = row_max
            lse[q_idx, q_head_idx] = cutlass.Float32.inf
            for d_v in cutlass.range(D_V):
                out[q_idx, q_head_idx, d_v] = cutlass.BFloat16(0.0)
            return

        # Pass 2: exp(QK - max), which is the denominator before attn_sink.
        normalizer = cutlass.Float32(0.0)
        for token_slot in cutlass.range(topk):
            token_idx = indices[q_idx, kv_head_idx, token_slot]
            if token_idx >= 0 and token_idx < kv.shape[0]:
                score = cutlass.Float32(0.0)
                for d in cutlass.range(D_QK):
                    score += cutlass.Float32(q[q_idx, q_head_idx, d]) * cutlass.Float32(
                        kv[token_idx, kv_head_idx, d]
                    )
                normalizer += cute.math.exp2(
                    (score - row_max) * sm_scale_log2, fastmath=True
                )

        # phase1.cuh stores lse without attn_sink but includes attn_sink when
        # normalizing O.
        max_logits[q_idx, q_head_idx] = row_max * sm_scale
        lse[q_idx, q_head_idx] = row_max * sm_scale + _math.log(normalizer)
        sink = cute.math.exp2(
            (attn_sink[q_head_idx] - row_max * sm_scale) * cutlass.Float32(LOG2_E),
            fastmath=True,
        )
        output_normalizer = normalizer + sink

        # Pass 3: P @ V.  Values occupy the first 512 elements of each
        # 576-element KV record, exactly as in the C++ phase1 kernel.
        for d_v in cutlass.range(D_V):
            value_sum = cutlass.Float32(0.0)
            for token_slot in cutlass.range(topk):
                token_idx = indices[q_idx, kv_head_idx, token_slot]
                if token_idx >= 0 and token_idx < kv.shape[0]:
                    score = cutlass.Float32(0.0)
                    for d in cutlass.range(D_QK):
                        score += cutlass.Float32(q[q_idx, q_head_idx, d]) * cutlass.Float32(
                            kv[token_idx, kv_head_idx, d]
                        )
                    probability = cute.math.exp2(
                        (score - row_max) * sm_scale_log2, fastmath=True
                    )
                    value_sum += probability * cutlass.Float32(
                        kv[token_idx, kv_head_idx, d_v]
                    )
            out[q_idx, q_head_idx, d_v] = value_sum / output_normalizer


def _as_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    """Expose a contiguous CUDA tensor to CuTeDSL without copying it."""
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("CuTeDSL inputs must be contiguous CUDA tensors")
    return from_dlpack(tensor, assumed_align=16)


def sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor,
    *,
    threads_per_block: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the independent CuTeDSL sparse-prefill reference.

    Returns ``(out, max_logits, lse)`` in exactly the same order as
    ``flash_mla_sparse_fwd``.  This function intentionally does not import
    ``flash_mla`` so it can be used to compare against the existing extension.
    """
    if q.dtype is not torch.bfloat16 or kv.dtype is not torch.bfloat16:
        raise TypeError("q and kv must be torch.bfloat16")
    if indices.dtype is not torch.int32 or attn_sink.dtype is not torch.float32:
        raise TypeError("indices must be int32 and attn_sink must be float32")
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("q, kv, and indices must all be rank-3 tensors")
    if q.shape[2] != D_QK or kv.shape[2] != D_QK:
        raise ValueError("the initial CuTeDSL port supports d_qk=576 only")
    if q.shape[0] != indices.shape[0] or kv.shape[1] != indices.shape[1]:
        raise ValueError("incompatible q/kv/indices shapes")
    if attn_sink.shape != (q.shape[1],):
        raise ValueError("attn_sink must have shape [h_q]")
    if not (q.is_cuda and kv.is_cuda and indices.is_cuda and attn_sink.is_cuda):
        raise ValueError("all inputs must be CUDA tensors")

    out = torch.empty((q.shape[0], q.shape[1], D_V), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
    lse = torch.empty_like(max_logits)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)

    op = Sm90SparsePrefillCuTeDsl(threads_per_block)
    compiled = cute.compile(
        op,
        _as_cute_tensor(q),
        _as_cute_tensor(kv),
        _as_cute_tensor(indices),
        _as_cute_tensor(attn_sink),
        _as_cute_tensor(out),
        _as_cute_tensor(max_logits),
        _as_cute_tensor(lse),
        sm_scale,
        stream,
    )
    compiled(
        _as_cute_tensor(q),
        _as_cute_tensor(kv),
        _as_cute_tensor(indices),
        _as_cute_tensor(attn_sink),
        _as_cute_tensor(out),
        _as_cute_tensor(max_logits),
        _as_cute_tensor(lse),
        sm_scale,
        stream,
    )
    return out, max_logits, lse
