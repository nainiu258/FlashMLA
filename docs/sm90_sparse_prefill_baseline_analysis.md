# SM90 Sparse Prefill Baseline：代码与流水线分析

本文只描述当前 SM90 sparse-prefill **baseline**：
`csrc/sm90/prefill/sparse/phase1.cuh` 中由
`run_fwd_phase1_kernel<D_QK, HAVE_TOPK_LENGTH>()` 启动的
`KernelTemplate<..., false, false>` 路径。文末会说明 DSMEM crossover 与 baseline
的区别，但不把它混入 baseline 的数据路径。

## 1. 调用链与输入

```text
Python: flash_mla.flash_mla_sparse_fwd(...)
  -> pybind: sparse_prefill_fwd
  -> csrc/api/sparse_fwd.h
  -> sm90::fwd::run_fwd_phase1_kernel<D_QK, HAVE_TOPK_LENGTH>
  -> KernelTemplate::run
  -> sparse_attn_fwd_kernel
  -> KernelTemplate::devfunc
```

主要张量语义如下。

```text
Q       [s_q, h_q, d_qk]，BF16
KV      [s_kv, 1, d_qk]，BF16 MLA latent；不是分开的全局 K/V 张量
indices [s_q, topk]，给出每个 Q token 选中的 KV token
O       [s_q, h_q, 512]，BF16
```

`d_qk` 支持 512 或 576；`d_v=512`。对 `d_qk=576`，KV latent 的前 512
维提供 value/nope 部分，最后 64 维是 RoPE 部分。kernel 内的 `K0/K1` 不是 K
和 V 的区别，而是连续处理的两个 64-token top-k slot。

## 2. Grid 与 CTA 工作范围

baseline 常量位于 `config.h`：

```text
B_H       = 64 heads / CTA
B_TOPK    = 64 tokens / sparse KV block
threads   = 384 = 3 warpgroups × 128 threads
cluster   = 1 CTA
```

launch grid 为：

```text
grid.x = s_q * (h_q / 64)
block  = 384 threads
```

CTA 坐标计算：

```cpp
q_h_idx = blockIdx.x % (h_q / B_H);
s_q_idx = blockIdx.x / (h_q / B_H);
```

因此一个 CTA 固定处理：

```text
一个 Q token × 64 个 query heads × 该 Q 的全部 top-k KV blocks
```

若 `h_q=128`，同一个 Q token 对应两个**独立** CTA：一个负责 head `0..63`，
另一个负责 `64..127`。baseline 没有 cluster/DSMEM，也没有 CTA 间同步；两个 CTA
各自加载自己需要的完整 top-k KV。

## 3. CTA 内角色

```text
WG0 (128 threads): K0 的 QK、P0 softmax、P0×V0L，以及 P1×V1L
WG1 (128 threads): K1 的 QK、P1 softmax、P1×V1R，以及 P0×V0R
WG2 (128 threads): producer，读取 indices 并填充双 KV slot
```

所有 warpgroup 访问同一 CTA 的 shared memory。`rP`、`rS`、`rO` 等是各自的
register fragment；跨 WG 需要共享的概率矩阵会暂存 shared memory。

## 4. Q：规则 TMA load

Q 是规则二维 tile `[64 heads, D_Q]`。host 端通过 CuTe
`make_tma_copy(SM90_TMA_LOAD, ...)` 建立 descriptor；CTA 中一个 elected thread
选择本 CTA 的 tile 后发起：

```text
global Q -> TMA load -> shared-memory sQ
```

完成由 `bar_q` 通知，WG0/WG1 均在消费 Q 前等待它。Q 在一个 CTA 生命周期中只加载
一次，后续所有 top-k block 重用这份 `sQ`；它没有随 KV block 双缓冲。

对 `D_Q=576`，Q tile 大小为 `64 × 576 × 2 B = 72 KiB`。

## 5. KV：稀疏 cp.async gather，而非 TMA load

KV 每一行的地址由 `indices` 决定：

```text
kv_addr(i, col) = kv_base + indices[i] * stride_kv + col
```

这不是单个仿射 tensor tile，所以 baseline 不能用一个普通 TMA descriptor 直接搬完整
KV block。WG2 按 token index 计算动态地址，调用
`cp_async_cacheglobal_l2_prefetch_256B(...)`：

```text
global KV -> cp.async -> CTA local shared memory
```

它同时请求 L2 256B prefetch hint。K/V slot 填充完后，通过 mbarrier 通知消费者。

## 6. 两 block 双缓冲与四段交织加载

producer 一次管理两个 64-token top-k block：`K0` 与 `K1`。每个 block 按 value
output 的左右半部拆成四段，交织的 producer 顺序为：

```text
1. K0 left   (V0L) -> K0-ready-left
2. K1 right  (V1R) -> K1-ready-right
3. K0 right  (V0R) -> K0-ready-right
4. K1 left   (V1L) -> K1-ready-left
```

目的不是 K 和 V 的混淆，而是让消费者按最早可用的半块开始计算：

```text
WG0 先做 K0 的 QK，再消费 V0L
WG1 先做 K1 的 QK，再消费 V1R
随后交换概率，WG0 消费 V1L，WG1 消费 V0R
```

`bar_k*_ready` 表示 producer 已填完可消费的 slot/半段；`bar_k*_free` 表示
consumer 的 GMMA 已完成、该存储可被 producer 覆盖。两个 phase 交替，防止 producer
覆盖仍被 QK/PV 使用的 buffer。

## 7. 单个 CTA 的计算与通信

对于一对 top-k block，记：

```text
P0 = softmax(Q K0^T)
P1 = softmax(Q K1^T)
```

步骤如下：

```text
WG0: QK(K0) -> online softmax(P0) -> rS(P0) 写入 sS0
WG1: QK(K1) -> 等待 WG0 的 max 状态 -> online softmax(P1) -> rS(P1) 写入 sS1

WG0 local : P0 × V0L
WG1 local : P1 × V1R

WG0 remote: sS1(P1) × V1L
WG1 remote: sS0(P0) × V0R
```

因此两侧的局部 output 是：

```text
O_L = P0 V0L + P1 V1L
O_R = P0 V0R + P1 V1R
```

`P0/P1` 的交换经 shared memory (`sS0/sS1`) 加 named barrier 完成；V 不需要
“交换”，因为 V slot 本来就在 CTA shared memory，两个 consumer WG 都能读取。
online softmax 的 `m/l` 状态在两个 WG 间以 shared-memory scratch 和 named barrier
协调。PV 累积 `rO` 主要留在寄存器；最终只对必要部分做 WG 间 reduction、归一化，再
通过 TMA store 写回 O。CTA 不会在计算中与其它 CTA 同步。

## 8. 流水线概览

```text
WG2 producer:  load K0L -> load K1R -> load K0R -> load K1L -> next pair
                     |          |           |           |
WG0 consumer : wait K0L/R -> QK0 -> softmax P0 -> P0V0L -> P1V1L
WG1 consumer : wait K1R/L -> QK1 -> softmax P1 -> P1V1R -> P0V0R
                     \__________ P0/P1 via CTA shared memory __________/
```

在 steady state，下一对 K 的 producer copy 与当前对的 QK/PV/softmax 重叠；双 buffer
使 producer 与 consumer 不会争用同一 KV slot。

## 9. 资源与性能含义

`D_QK=576` 时，该 CTA 的 dynamic shared memory 约为 225.5 KiB（加上静态开销约
226.5 KiB）。结合约 384 threads 和高寄存器使用，Hopper 上该配置通常只能实现
**1 CTA / SM**。

这不意味着 CTA 调度本身有类似 CPU context switch 的巨大成本。一个 CTA 结束后，硬件
会调度下一个 block；但前一个 CTA 的 registers/shared memory 状态不会保留，新 CTA 仍
要重新加载 Q、填充 KV pipeline。性能重点通常是：

```text
1. sparse KV gather 的 L2/HBM 延迟与带宽
2. producer/consumer barrier 等待
3. 只有一个 resident CTA 时的 latency hiding 能力
4. 不同 Q 的 top-k 长度与 indices 导致的负载不均
```

以 `topk=2048, D_Q=576` 为例：每 CTA 需要 32 个 64-token KV block，KV 流量远大于
一次 72 KiB Q load。因此这里“围绕 Q 并行”是指用 `(Q token, head-group)` 提供大量
独立 CTA 来填满 GPU，并不是指 Q load 是主要成本。

## 10. 与 sparse decode 的关系

两者都做 QK、online softmax 和 PV，但调度/数据复用不同：

```text
SM90 prefill baseline:
  CTA = (Q token, 64 heads)
  h_q=128 时两个 CTA 独立，各自读取完整 sparse KV
  grid 用 s_q × head-group 提供并行度

SM90 sparse decode:
  h_q=128 时两个 CTA 构成 cluster 共同处理一个 request
  两个 CTA 分担 KV token，再通过 DSMEM 共享
  scheduler 可让同一常驻 cluster 串行处理多个 request
```

decode cluster 串行处理多个 request 并不代表跨 request 复用 Q、KV 或 softmax accumulator；
每个 request 仍需独立加载/初始化。其主要作用是 persistent scheduling 和 split-KV/load
balance。真正的 KV global-load 复用发生在同一 request 的两个 head-group CTA 间。

## 11. 非 baseline：DSMEM crossover 实验

`run_fwd_phase1_kernel_dsmem_crossover<576, false, true>` 是实验路径，不是上述 baseline。
它将 `h_q=128` 的两 CTA 组成 size-2 cluster：每 CTA 只用普通 `ld.global` 加载自己负责的
32 个 token 的 16B words，写入本地 shared memory，并用：

```text
st.async.shared::cluster.mbarrier::complete_tx::bytes
```

把同一 word 异步写入 peer CTA 的 DSMEM。这样两个 CTA 合起来只从 global memory 读取一次
完整 64-token block，但引入 register load-use latency、DSMEM 通信与 cluster 同步。

peer 的 mbarrier 由 `arrive_and_expect_tx(bytes)` 预置，`st.async` 完成时自动
`complete_tx`；本地普通 store 则用 `fence_view_async_shared()` 后普通 `bar.arrive()`
发布可见性。它是否快于 baseline 取决于 L2/HBM miss、DSMEM 开销和 occupancy，必须 profile。

