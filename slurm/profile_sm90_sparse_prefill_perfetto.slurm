#!/usr/bin/env bash
# Emit a Perfetto trace of one SM90 BF16 sparse-prefill CTA's pipeline.  Open
# the .perfetto-trace at https://ui.perfetto.dev.
#
# Submit from the repository root (build first, then run):
#   sbatch slurm/build_sm90_sparse_prefill_perfetto.slurm
#   sbatch slurm/profile_sm90_sparse_prefill_perfetto.slurm
#
# This job never builds the CUDA extension; it relies on the
# FLASH_MLA_TRACE_PERFETTO=1 build produced by the build script above.
#
# Default case matches the existing NCU report: sq=128, skv=65536, topk=2048,
# hq=128, dqk=576. TRACE_CTA must select one CTA: the trace uses private,
# atomic-free regions for producer/WG0/WG1 so it does not perturb their overlap.

#SBATCH --job-name=mla-sp-perfetto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/mla_sp_perfetto_%j.out
#SBATCH --error=slurm/logs/mla_sp_perfetto_%j.err

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
WARMUP="${WARMUP:-1}"
TRACE_CTA="${TRACE_CTA:-0}"

if (( TOPK <= 0 || TOPK % 128 != 0 )); then
  echo "ERROR: TOPK must be a positive multiple of 128, got $TOPK" >&2
  exit 2
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-unset}"

# The trace API only exists in a FLASH_MLA_TRACE_PERFETTO=1 build.  Fail fast
# with a clear message if the extension has not been built at all (the run
# itself TORCH_CHECKs with "rebuild with FLASH_MLA_TRACE_PERFETTO=1" if a
# non-trace build is present).
shopt -s nullglob
extension_files=(flash_mla/cuda*.so)
shopt -u nullglob
if (( ${#extension_files[@]} == 0 )); then
  echo "ERROR: no built extension found at flash_mla/cuda*.so" >&2
  echo "Run slurm/build_sm90_sparse_prefill_perfetto.slurm successfully first." >&2
  exit 4
fi
echo "using build artifact: ${extension_files[*]}"

JOB_ID="${SLURM_JOB_ID:-local}"
JOB_TAG="sq${SQ}_skv${SKV}_topk${TOPK}_hq${HQ}_dqk${DQK}_cta${TRACE_CTA}_${JOB_ID}"
OUT_TRACE="slurm/profiles/mla_sp_perfetto_${JOB_TAG}.perfetto-trace"

echo "case: s_q=$SQ s_kv=$SKV topk=$TOPK h_q=$HQ d_qk=$DQK warmup=$WARMUP trace_only_cta=$TRACE_CTA"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" tests/profile_sm90_sparse_prefill_perfetto.py \
  --s-q "$SQ" --s-kv "$SKV" --topk "$TOPK" --h-q "$HQ" --d-qk "$DQK" \
  --warmup "$WARMUP" --trace-only-cta "$TRACE_CTA" --out "$OUT_TRACE"

if [[ ! -f "$OUT_TRACE" ]]; then
  echo "ERROR: trace was not produced at $OUT_TRACE" >&2
  exit 3
fi

echo "Perfetto trace: $OUT_TRACE"
echo "Open it at https://ui.perfetto.dev"
