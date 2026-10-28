#!/usr/bin/env bash
# Profile one SM90 sparse-decode kernel with Nsight Compute.
#
# The decode path has a split-KV main kernel and, if needed, a separate combine
# kernel. Profile them separately so their metrics are not conflated:
#   sbatch slurm/profile_sm90_sparse_decode_ncu.slurm
#   KERNEL=combine sbatch slurm/profile_sm90_sparse_decode_ncu.slurm
#
# Useful overrides:
#   BATCH_SIZE=64 SQ=2 SKV=32768 TOPK=2048 HQ=128 DQK=576 \
#     KERNEL=splitkv STALL_ANALYSIS=1 sbatch slurm/profile_sm90_sparse_decode_ncu.slurm

#SBATCH --job-name=mla-sd-ncu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/mla_sd_ncu_%j.out
#SBATCH --error=slurm/logs/mla_sd_ncu_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
NCU_BIN="${NCU_BIN:-ncu}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if ! command -v "$NCU_BIN" >/dev/null 2>&1; then
  echo "ERROR: Nsight Compute was not found: $NCU_BIN" >&2
  exit 2
fi
NCU_BIN="$(command -v "$NCU_BIN")"

shopt -s nullglob
EXTENSION_ARTIFACTS=(flash_mla/cuda*.so)
if (( ${#EXTENSION_ARTIFACTS[@]} == 0 )); then
  echo "ERROR: No built FlashMLA extension found (flash_mla/cuda*.so)." >&2
  echo "Run: sbatch slurm/build_sm90_sparse_prefill_dsmem.slurm" >&2
  exit 2
fi

mkdir -p slurm/logs slurm/profiles

BATCH_SIZE="${BATCH_SIZE:-128}"
SQ="${SQ:-2}"
SKV="${SKV:-32768}"
TOPK="${TOPK:-2048}"
HQ="${HQ:-128}"
DQK="${DQK:-576}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
WARMUP="${WARMUP:-300}"
STALL_ANALYSIS="${STALL_ANALYSIS:-0}"
KERNEL="${KERNEL:-splitkv}"

case "$KERNEL" in
  splitkv)
    KERNEL_REGEX='.*flash_fwd_splitkv_mla_fp8_sparse_kernel.*'
    ;;
  combine)
    KERNEL_REGEX='.*flash_fwd_mla_combine_kernel.*'
    ;;
  *)
    echo "ERROR: KERNEL must be splitkv or combine, got $KERNEL" >&2
    exit 3
    ;;
esac

if (( TOPK <= 0 || TOPK > SKV || TOPK % 64 != 0 )); then
  echo "ERROR: TOPK must be in (0, SKV] and a multiple of 64" >&2
  exit 3
fi

JOB_ID="${SLURM_JOB_ID:-local}"
JOB_TAG="${KERNEL}_b${BATCH_SIZE}_sq${SQ}_skv${SKV}_topk${TOPK}_hq${HQ}_dqk${DQK}_${JOB_ID}"
NCU_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/flashmla_ncu_${JOB_ID}}"
mkdir -p "$NCU_TMPDIR"
REPORT_STEM="$NCU_TMPDIR/mla_sd_${JOB_TAG}"
FINAL_REPORT="slurm/profiles/mla_sd_${JOB_TAG}.ncu-rep"
FINAL_CSV="slurm/logs/mla_sd_${JOB_TAG}.csv"

echo "host=$(hostname) job=$JOB_ID gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "case: kernel=$KERNEL b=$BATCH_SIZE s_q=$SQ s_kv=$SKV topk=$TOPK h_q=$HQ d_qk=$DQK"
echo "warmup=$WARMUP stall_analysis=$STALL_ANALYSIS"

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
SECTIONS=(SpeedOfLight ComputeWorkloadAnalysis MemoryWorkloadAnalysis SchedulerStats)
if [[ "$STALL_ANALYSIS" == "1" ]]; then
  SECTIONS+=(WarpStateStats SourceCounters InstructionStats)
fi
SECTION_ARGS=()
for section in "${SECTIONS[@]}"; do
  SECTION_ARGS+=(--section "$section")
done

# --launch-skip counts matching kernels only, so the final call contributes
# exactly one launch of the selected split-KV or combine kernel.
sudo -E "$NCU_BIN" \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "regex:${KERNEL_REGEX}" \
  --launch-skip "$WARMUP" \
  --launch-count 1 \
  "${SECTION_ARGS[@]}" \
  --metrics "$METRIC_CSV" \
  --csv \
  --page raw \
  --export "$REPORT_STEM" \
  "$PYTHON_BIN" tests/profile_sm90_sparse_decode.py \
    --batch-size "$BATCH_SIZE" --s-q "$SQ" --s-kv "$SKV" --topk "$TOPK" \
    --h-q "$HQ" --d-qk "$DQK" --block-size "$BLOCK_SIZE" --warmup "$WARMUP" \
  | tee "$FINAL_CSV"

cp "$REPORT_STEM.ncu-rep" "$FINAL_REPORT"
echo "Saved NCU report: $FINAL_REPORT"
echo "Saved NCU CSV/stdout: $FINAL_CSV"
