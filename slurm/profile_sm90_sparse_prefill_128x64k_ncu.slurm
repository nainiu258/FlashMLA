#!/usr/bin/env bash
# Profile one SM90 BF16 sparse-prefill kernel with Nsight Compute.
#
# Default case: an existing 64K KV cache plus a 128-token prefill chunk.
# The sparse kernel gathers TOPK tokens per new query; it does not scan all
# SKV tokens.  The default has the following requested-traffic model:
#   KV=576 MiB, indices=2 MiB, Q=18 MiB, O=16 MiB, stats=128 KiB.
#
# Submit from the repository root:
#   sbatch slurm/profile_sm90_sparse_prefill_128x64k_ncu.slurm
#
# Optional overrides:
#   TOPK=4096 WARMUP=300 sbatch slurm/profile_sm90_sparse_prefill_128x64k_ncu.slurm
#
# In addition to the explicit metrics below, this job collects SpeedOfLight,
# ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, and SchedulerStats.

#SBATCH --job-name=mla-sp-128x64k-ncu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/mla_sp_128x64k_ncu_%j.out
#SBATCH --error=slurm/logs/mla_sp_128x64k_ncu_%j.err

set -euo pipefail

# Slurm may run a spooled copy under /var/spool/slurmd, so use the submit
# directory rather than $0 to find the checkout.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi

NCU_BIN="${NCU_BIN:-ncu}"
if ! command -v "$NCU_BIN" >/dev/null 2>&1; then
  echo "ERROR: Nsight Compute was not found: $NCU_BIN" >&2
  exit 2
fi

mkdir -p slurm/logs slurm/profiles

SQ="${SQ:-128}"
SKV="${SKV:-65536}"
TOPK="${TOPK:-2048}"
HQ="${HQ:-128}"
DQK="${DQK:-576}"
WARMUP="${WARMUP:-300}"
# Set to 1 to attribute scheduler stalls to source lines. This adds replay
# passes, so leave it off for the regular throughput profile.
STALL_ANALYSIS="${STALL_ANALYSIS:-0}"

if (( TOPK <= 0 || TOPK % 128 != 0 )); then
  echo "ERROR: TOPK must be a positive multiple of 128, got $TOPK" >&2
  exit 2
fi
if (( TOPK > SKV )); then
  echo "ERROR: TOPK ($TOPK) must not exceed SKV ($SKV) for this all-valid profile" >&2
  exit 2
fi
JOB_ID="${SLURM_JOB_ID:-local}"
JOB_TAG="sq${SQ}_skv${SKV}_topk${TOPK}_hq${HQ}_dqk${DQK}_${JOB_ID}"

# NCU report creation is more reliable on node-local storage than on NFS.
NCU_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/flashmla_ncu_${JOB_ID}}"
mkdir -p "$NCU_TMPDIR"
REPORT_STEM="$NCU_TMPDIR/mla_sp_128x64k_${JOB_TAG}"
FINAL_REPORT="slurm/profiles/mla_sp_128x64k_${JOB_TAG}.ncu-rep"
FINAL_CSV="slurm/logs/mla_sp_128x64k_${JOB_TAG}.csv"

echo "host=$(hostname) job=$JOB_ID gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "case: s_q=$SQ s_kv=$SKV topk=$TOPK h_q=$HQ d_qk=$DQK"
echo "NCU direct mode: warmup=$WARMUP; no sudo or sudo -E."
echo "stall_analysis=$STALL_ANALYSIS"
echo "requested-byte model excludes L2 reuse; compare it with NCU DRAM counters."

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

# These sections provide the compute/memory utilization and warp-scheduler
# analyses in the .ncu-rep report. They require additional replay passes.
SECTIONS=(
  SpeedOfLight
  ComputeWorkloadAnalysis
  MemoryWorkloadAnalysis
  SchedulerStats
)
if [[ "$STALL_ANALYSIS" == "1" ]]; then
  # setup.py builds with -lineinfo. These sections attribute sampled warp
  # stalls/instructions to source lines, separating barrier waits from GMMA
  # waits and memory scoreboards, which SchedulerStats cannot do alone.
  SECTIONS+=(
    WarpStateStats
    SourceCounters
    InstructionStats
  )
fi
SECTION_ARGS=()
for section in "${SECTIONS[@]}"; do
  SECTION_ARGS+=(--section "$section")
done

# The Python entry point emits WARMUP matching kernels, then one final kernel.
# Restrict NCU to that final sparse_attn_fwd_kernel launch.
sudo -E /usr/local/cuda-13.0/bin/ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*sparse_attn_fwd_kernel.*' \
  --launch-skip "$WARMUP" \
  --launch-count 1 \
  "${SECTION_ARGS[@]}" \
  --metrics "$METRIC_CSV" \
  --csv \
  --page raw \
  --export "$REPORT_STEM" \
  "$PYTHON_BIN" tests/profile_sm90_sparse_prefill.py \
    --s-q "$SQ" --s-kv "$SKV" --topk "$TOPK" --h-q "$HQ" --d-qk "$DQK" --warmup "$WARMUP" \
    --regenerate-input-per-launch \
  | tee "$FINAL_CSV"

cp "$REPORT_STEM.ncu-rep" "$FINAL_REPORT"
echo "Saved NCU report: $FINAL_REPORT"
echo "Saved NCU CSV/stdout: $FINAL_CSV"
