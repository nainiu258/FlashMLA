#!/usr/bin/env bash
# Build the clock64-instrumented SM90 sparse-prefill kernel and measure its
# producer-ready mbarrier waits.  The default case matches the existing NCU
# report: sq=128, skv=65536, topk=2048, hq=128, dqk=576.

#SBATCH --job-name=mla-sp-ready-wait
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/mla_sp_ready_wait_%j.out
#SBATCH --error=slurm/logs/mla_sp_ready_wait_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p slurm/logs
SQ="${SQ:-128}"
SKV="${SKV:-65536}"
TOPK="${TOPK:-2048}"
HQ="${HQ:-128}"
DQK="${DQK:-576}"
WARMUP="${WARMUP:-300}"

if (( TOPK <= 0 || TOPK % 128 != 0 )); then
  echo "ERROR: TOPK must be a positive multiple of 128, got $TOPK" >&2
  exit 2
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "case: s_q=$SQ s_kv=$SKV topk=$TOPK h_q=$HQ d_qk=$DQK warmup=$WARMUP"
echo "Building with FLASH_MLA_PROFILE_READY_WAIT=1"
FLASH_MLA_PROFILE_READY_WAIT=1 PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" setup.py build_ext --inplace

PYTHONUNBUFFERED=1 "$PYTHON_BIN" tests/profile_sm90_sparse_prefill.py \
  --s-q "$SQ" --s-kv "$SKV" --topk "$TOPK" --h-q "$HQ" --d-qk "$DQK" \
  --warmup "$WARMUP" --ready-wait-cycles
