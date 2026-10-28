#!/usr/bin/env bash
# Build the SM90 BF16 sparse-prefill DSMEM crossover extension.
#
# Submit from the repository root:
#   sbatch slurm/build_sm90_sparse_prefill_dsmem.slurm
# Run the benchmark after this job completes successfully:
#   sbatch slurm/bench_sm90_sparse_prefill_dsmem.slurm

#SBATCH --job-name=mla-sp-prefill-dsmem-build
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/mla_sp_prefill_dsmem_build_%j.out
#SBATCH --error=slurm/logs/mla_sp_prefill_dsmem_build_%j.err

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
# stale translation units are rebuilt. Set BUILD_MODE=full for a clean rebuild.
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
echo "build: experimental SM90 BF16 prefill DSMEM crossover (mode=$BUILD_MODE)"
# Do not touch sources here: modifying timestamps would invalidate an
# otherwise reusable incremental build.
PYTHONUNBUFFERED=1 "$PYTHON_BIN" setup.py build_ext --inplace "${BUILD_ARGS[@]}"

shopt -s nullglob
extension_files=(flash_mla/cuda*.so)
shopt -u nullglob
if (( ${#extension_files[@]} == 0 )); then
  echo "ERROR: build completed without producing flash_mla/cuda*.so" >&2
  exit 5
fi
echo "build artifact: ${extension_files[*]}"
