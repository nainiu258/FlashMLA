#!/usr/bin/env bash
# Build the %globaltimer-instrumented SM90 BF16 sparse-prefill extension for
# Perfetto tracing.
#
# Submit from the repository root:
#   sbatch slurm/build_sm90_sparse_prefill_perfetto.slurm
# Then emit a trace:
#   sbatch slurm/profile_sm90_sparse_prefill_perfetto.slurm
#
# Set FLASH_MLA_DISABLE_SM100=1 on the sbatch command line to skip SM100 and
# only build the SM90 target the trace needs (faster):
#   FLASH_MLA_DISABLE_SM100=1 sbatch slurm/build_sm90_sparse_prefill_perfetto.slurm

#SBATCH --job-name=mla-sp-perfetto-build
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/mla_sp_perfetto_build_%j.out
#SBATCH --error=slurm/logs/mla_sp_perfetto_build_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: expected Python interpreter does not exist: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p slurm/logs

# `incremental` preserves Ninja's object/dependency state: only missing or
# stale translation units are rebuilt. Set BUILD_MODE=full for a clean rebuild
# (recommended when switching feature flags, e.g. from a non-trace build).
BUILD_MODE="${BUILD_MODE:-incremental}"
case "$BUILD_MODE" in
  incremental)
    BUILD_ARGS=()
    ;;
  full)
    BUILD_ARGS=(--force)
    ;;
  *)
    echo "ERROR: BUILD_MODE must be incremental or full, got $BUILD_MODE" >&2
    exit 4
    ;;
esac

echo "host=$(hostname) job=${SLURM_JOB_ID} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "build: SM90 BF16 sparse prefill, FLASH_MLA_TRACE_PERFETTO=1 (mode=$BUILD_MODE)"

FLASH_MLA_TRACE_PERFETTO=1 PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" setup.py build_ext --inplace "${BUILD_ARGS[@]}"

shopt -s nullglob
extension_files=(flash_mla/cuda*.so)
shopt -u nullglob
if (( ${#extension_files[@]} == 0 )); then
  echo "ERROR: build completed without producing flash_mla/cuda*.so" >&2
  exit 5
fi
echo "build artifact: ${extension_files[*]}"
