#!/usr/bin/env bash
# Trace the SM90 BF16 sparse-prefill launch with Nsight Systems.
#
# Submit from the repository root:
#   sbatch slurm/profile_sm90_sparse_prefill_128x64k_nsys.slurm
#
# The default uses WARMUP=0 so the trace contains exactly one
# sparse_attn_fwd_kernel launch. Set WARMUP>0 when a warm-cache timeline is
# useful; those warmup launches remain visible in the Nsight Systems report.
#
# Examples:
#   WARMUP=3 sbatch slurm/profile_sm90_sparse_prefill_128x64k_nsys.slurm
#   SQ=256 TOPK=4096 sbatch slurm/profile_sm90_sparse_prefill_128x64k_nsys.slurm

#SBATCH --job-name=mla-sp-128x64k-nsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/mla_sp_128x64k_nsys_%j.out
#SBATCH --error=slurm/logs/mla_sp_128x64k_nsys_%j.err

set -euo pipefail

# Slurm may execute a spooled copy of this script, so locate the checkout via
# the submit directory rather than relative to $0.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi

NSYS_BIN="${NSYS_BIN:-nsys}"
if ! command -v "$NSYS_BIN" >/dev/null 2>&1; then
  echo "ERROR: Nsight Systems was not found: $NSYS_BIN" >&2
  exit 2
fi

mkdir -p slurm/logs slurm/profiles

SQ="${SQ:-128}"
SKV="${SKV:-65536}"
TOPK="${TOPK:-2048}"
HQ="${HQ:-128}"
DQK="${DQK:-576}"
WARMUP="${WARMUP:-0}"

# Keep the default trace lightweight and focused on CUDA submission/execution.
# Enable memory accounting only when allocation lifetime is part of the issue.
CUDA_MEMORY_USAGE="${CUDA_MEMORY_USAGE:-false}"

if (( SQ <= 0 || SKV <= 0 || TOPK <= 0 )); then
  echo "ERROR: SQ, SKV, and TOPK must be positive" >&2
  exit 2
fi
if (( TOPK % 128 != 0 )); then
  echo "ERROR: TOPK must be a positive multiple of 128, got $TOPK" >&2
  exit 2
fi
if (( TOPK > SKV )); then
  echo "ERROR: TOPK ($TOPK) must not exceed SKV ($SKV)" >&2
  exit 2
fi
if (( WARMUP < 0 )); then
  echo "ERROR: WARMUP must be non-negative, got $WARMUP" >&2
  exit 2
fi

JOB_ID="${SLURM_JOB_ID:-local}"
JOB_TAG="sq${SQ}_skv${SKV}_topk${TOPK}_hq${HQ}_dqk${DQK}_${JOB_ID}"

# Write on node-local storage first. This avoids tracing I/O onto the shared
# filesystem, then preserves the portable .nsys-rep beside the NCU reports.
NSYS_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER:-unknown}/flashmla_nsys_${JOB_ID}}"
mkdir -p "$NSYS_TMPDIR"
REPORT_STEM="$NSYS_TMPDIR/mla_sp_128x64k_${JOB_TAG}"
FINAL_REPORT="slurm/profiles/mla_sp_128x64k_${JOB_TAG}.nsys-rep"

echo "host=$(hostname) job=$JOB_ID gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "case: s_q=$SQ s_kv=$SKV topk=$TOPK h_q=$HQ d_qk=$DQK warmup=$WARMUP"
echo "Nsight Systems trace: cuda,nvtx,osrt; CPU sampling disabled."
echo "Open the result with: nsys-ui $FINAL_REPORT"

"$NSYS_BIN" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cuda-memory-usage="$CUDA_MEMORY_USAGE" \
  --force-overwrite=true \
  --output "$REPORT_STEM" \
  "$PYTHON_BIN" tests/profile_sm90_sparse_prefill.py \
    --s-q "$SQ" --s-kv "$SKV" --topk "$TOPK" --h-q "$HQ" --d-qk "$DQK" --warmup "$WARMUP" \
  | tee "slurm/logs/mla_sp_128x64k_${JOB_TAG}.nsys.log"

if [[ ! -f "$REPORT_STEM.nsys-rep" ]]; then
  echo "ERROR: Nsight Systems did not produce $REPORT_STEM.nsys-rep" >&2
  exit 3
fi

cp "$REPORT_STEM.nsys-rep" "$FINAL_REPORT"
echo "Saved Nsight Systems report: $FINAL_REPORT"
