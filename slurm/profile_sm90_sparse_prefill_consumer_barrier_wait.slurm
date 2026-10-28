#!/usr/bin/env bash
# Build the clock64-instrumented SM90 sparse-prefill kernel and measure the
# four named-barrier waits exchanged by consumer WG0 and WG1.

#SBATCH --job-name=mla-sp-consumer-barrier
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/mla_sp_consumer_barrier_%j.out
#SBATCH --error=slurm/logs/mla_sp_consumer_barrier_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p slurm/logs slurm/profiles
SQ="${SQ:-128}"
SKV="${SKV:-65536}"
TOPK="${TOPK:-2048}"
HQ="${HQ:-128}"
DQK="${DQK:-576}"
WARMUP="${WARMUP:-300}"
JOB_ID="${SLURM_JOB_ID:-local}"
JOB_TAG="sq${SQ}_skv${SKV}_topk${TOPK}_hq${HQ}_dqk${DQK}_${JOB_ID}"
TRACE_FILE="slurm/profiles/mla_sp_consumer_barrier_${JOB_TAG}.pt"
SASS_FILE="slurm/profiles/mla_sp_consumer_barrier_${JOB_TAG}.cuobjdump.sass"

if (( TOPK <= 0 || TOPK % 128 != 0 )); then
  echo "ERROR: TOPK must be a positive multiple of 128, got $TOPK" >&2
  exit 2
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "case: s_q=$SQ s_kv=$SKV topk=$TOPK h_q=$HQ d_qk=$DQK warmup=$WARMUP"
echo "Building with FLASH_MLA_PROFILE_CONSUMER_BARRIER_WAIT=1"
FLASH_MLA_PROFILE_CONSUMER_BARRIER_WAIT=1 FLASH_MLA_DISABLE_SOURCE_IN_PTX=1 FLASH_MLA_DISABLE_SM100=1 PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" setup.py build_ext --inplace --force

PYTHONUNBUFFERED=1 "$PYTHON_BIN" tests/profile_sm90_sparse_prefill.py \
  --s-q "$SQ" --s-kv "$SKV" --topk "$TOPK" --h-q "$HQ" --d-qk "$DQK" \
  --warmup "$WARMUP" --regenerate-input-per-launch --consumer-barrier-wait-cycles \
  --consumer-barrier-wait-output "$TRACE_FILE"

/usr/local/cuda-13.0/bin/cuobjdump --dump-sass flash_mla/cuda.cpython-312-x86_64-linux-gnu.so > "$SASS_FILE"
echo "Saved raw barrier trace: $TRACE_FILE"
echo "Saved instrumented SASS dump: $SASS_FILE"
echo "clock/barrier SASS locations:"
rg -n "SR_CLOCK|BAR\.SYNC\.DEFER_BLOCKING" "$SASS_FILE" | head -80
