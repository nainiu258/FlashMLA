#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch slurm/profile_sm90_sparse_prefill_bandwidth.sbatch
# Optional overrides:
#   SQ=1024 SKV=65536 TOPK=2048 HQ=128 DQK=576 sbatch slurm/profile_sm90_sparse_prefill_bandwidth.sbatch
#
# This job profiles one fused sparse-prefill launch. It does not claim a
# time-separated bandwidth for QK, softmax, and PV: those phases deliberately
# overlap. Instead it records hardware traffic for the full kernel and prints
# a requested-byte model split by producer/Q/epilogue.

#SBATCH --job-name=mla-sp-prefill-bw
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=mla_sp_prefill_bw_%j.out
#SBATCH --error=mla_sp_prefill_bw_%j.err

set -euo pipefail

# Slurm may execute a spooled copy of this script under /var/spool/slurmd, so
# $0 cannot be used to locate the checkout. The submit directory is stable.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: expected Python interpreter does not exist: $PYTHON_BIN" >&2
  exit 2
fi
mkdir -p slurm/profiles slurm/logs

SQ="${SQ:-4096}"
SKV="${SKV:-64532}"
TOPK="${TOPK:-2048}"
HQ="${HQ:-128}"
DQK="${DQK:-576}"
WARMUP="${WARMUP:-3}"
JOB_TAG="sq${SQ}_skv${SKV}_topk${TOPK}_hq${HQ}_dqk${DQK}_${SLURM_JOB_ID}"
# NCU report creation uses file operations that can fail on a shared/NFS work
# directory even when ordinary shell output works. Write the report on the
# compute node first, then copy the completed artifact back to the repository.
NCU_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/flashmla_ncu_${SLURM_JOB_ID}}"
mkdir -p "$NCU_TMPDIR"
REPORT_STEM="$NCU_TMPDIR/mla_sp_prefill_bw_${JOB_TAG}"
FINAL_REPORT="slurm/profiles/mla_sp_prefill_bw_${JOB_TAG}.ncu-rep"

echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "case: s_q=$SQ s_kv=$SKV topk=$TOPK h_q=$HQ d_qk=$DQK"

# Counter meanings:
# - DRAM: physical HBM traffic after cache effects.
# - L2: requests/traffic reaching L2, useful for evaluating random-KV reuse.
# - global LDGSTS: the producer's cp.async global-to-SMEM traffic category.
#   WGMMA SMEM operands are not fully represented by ordinary LSU byte counters,
#   so do not label an LSU metric as total shared-memory bandwidth.
METRICS=(
  gpu__time_duration.sum
  dram__bytes_read.sum
  dram__bytes_write.sum
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  lts__t_bytes.sum
  lts__t_requests.sum
  lts__t_requests_lookup_hit.sum
  lts__t_requests_lookup_miss.sum
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
  l1tex__t_bytes_pipe_lsu_mem_global_op_ldgsts_cache_access.sum
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum
)
METRIC_CSV="$(IFS=,; echo "${METRICS[*]}")"

# NCU matches only sparse_attn_fwd_kernel. --launch-skip excludes the three
# warmups emitted by the Python entry point; --launch-count limits collection
# to the final launch. The report is written to node-local storage first.
sudo -E ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sparse_attn_fwd_kernel.*' \
  --launch-skip "$WARMUP" \
  --launch-count 1 \
  --metrics "$METRIC_CSV" \
  --csv \
  --page raw \
  --export "$REPORT_STEM" \
  "$PYTHON_BIN" tests/profile_sm90_sparse_prefill.py \
    --s-q "$SQ" --s-kv "$SKV" --topk "$TOPK" --h-q "$HQ" --d-qk "$DQK" --warmup "$WARMUP" \
  | tee "slurm/logs/mla_sp_prefill_bw_${JOB_TAG}.csv"

cp "$REPORT_STEM.ncu-rep" "$FINAL_REPORT"
echo "Saved NCU report: $FINAL_REPORT"
echo "Saved CSV/stdout: slurm/logs/mla_sp_prefill_bw_${JOB_TAG}.csv"
