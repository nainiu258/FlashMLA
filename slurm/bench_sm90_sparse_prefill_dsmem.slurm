#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch slurm/build_sm90_sparse_prefill_dsmem.slurm
#   sbatch slurm/bench_sm90_sparse_prefill_dsmem.slurm
#
# This job never builds the CUDA extension. It uses the extension produced by
# build_sm90_sparse_prefill_dsmem.slurm in this shared checkout.

#SBATCH --job-name=mla-sp-prefill-dsmem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/mla_sp_prefill_dsmem_%j.out
#SBATCH --error=slurm/logs/mla_sp_prefill_dsmem_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: expected Python interpreter does not exist: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p slurm/logs

SQ="${SQ:-128}"
SKV="${SKV:-65536}"
TOPK="${TOPK:-2048}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-100}"

echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
shopt -s nullglob
extension_files=(flash_mla/cuda*.so)
shopt -u nullglob
if (( ${#extension_files[@]} == 0 )); then
  echo "ERROR: no built extension found at flash_mla/cuda*.so" >&2
  echo "Run slurm/build_sm90_sparse_prefill_dsmem.slurm successfully first." >&2
  exit 4
fi
echo "using build artifact: ${extension_files[*]}"
echo "benchmark: s_q=$SQ s_kv=$SKV topk=$TOPK"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" tests/bench_sm90_sparse_prefill_dsmem.py --s-q "$SQ" --s-kv "$SKV" --topk "$TOPK" --warmup "$WARMUP" --iters "$ITERS"
