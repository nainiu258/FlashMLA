#!/usr/bin/env bash
# CUDA-event fallback: no NCU and no GPU performance-counter permission needed.
# Submit from the repository root:
#   sbatch slurm/bench_sm90_sparse_prefill_event.slurm
# Optional overrides:
#   SQ=4096 SKV=65536 TOPK=2048 HQ=128 DQK=576 ITERS=50 \
#     sbatch slurm/bench_sm90_sparse_prefill_event.slurm

#SBATCH --job-name=mla-sp-prefill-event
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=mla_sp_prefill_event_%j.out
#SBATCH --error=mla_sp_prefill_event_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: expected Python interpreter does not exist: $PYTHON_BIN" >&2
  exit 2
fi
mkdir -p slurm/logs

SQ="${SQ:-4096}"
SKV="${SKV:-32768}"
TOPK="${TOPK:-2048}"
HQ="${HQ:-128}"
DQK="${DQK:-576}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
JOB_TAG="sq${SQ}_skv${SKV}_topk${TOPK}_hq${HQ}_dqk${DQK}_${SLURM_JOB_ID}"

echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "case: s_q=$SQ s_kv=$SKV topk=$TOPK h_q=$HQ d_qk=$DQK warmup=$WARMUP iters=$ITERS"
"$PYTHON_BIN" tests/profile_sm90_sparse_prefill.py \
  --s-q "$SQ" --s-kv "$SKV" --topk "$TOPK" --h-q "$HQ" --d-qk "$DQK" \
  --warmup "$WARMUP" --event-iters "$ITERS" \
  | tee "slurm/logs/mla_sp_prefill_event_${JOB_TAG}.log"
