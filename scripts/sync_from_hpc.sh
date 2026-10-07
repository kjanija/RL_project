#!/usr/bin/env bash
set -euo pipefail

# HPC cluster configuration (overrideable via environment variables)
REMOTE_HOST="${REMOTE_HOST:-"orfeo"}"
REMOTE_DIR="${REMOTE_DIR:-"/orfeo/scratch/dssc/kj_mersimoski/RL_project"}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ALL_WEIGHTS=false
LOGS_ONLY=false
EXTRA_RSYNC_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --all-weights)
      ALL_WEIGHTS=true
      ;;
    --logs-only)
      LOGS_ONLY=true
      ;;
    *)
      EXTRA_RSYNC_ARGS+=("$arg")
      ;;
  esac
done

remote_dir_exists() {
  local target_dir="$1"
  ssh "${REMOTE_HOST}" "[ -d '${target_dir}' ]" 2>/dev/null
}

remote_file_exists() {
  local target_file="$1"
  ssh "${REMOTE_HOST}" "[ -f '${target_file}' ]" 2>/dev/null
}

echo "=========================================================="
echo " Syncing Results from HPC (${REMOTE_HOST}:${REMOTE_DIR})"
echo " Mode: $(if [ "$ALL_WEIGHTS" = true ]; then echo "All Weights"; elif [ "$LOGS_ONLY" = true ]; then echo "Logs & Metrics Only"; else echo "Best/Final Models + Metrics"; fi)"
echo "=========================================================="

# 1. Slurm Job Logs
if remote_dir_exists "${REMOTE_DIR}/slurm_logs"; then
  echo "==> Syncing slurm_logs..."
  mkdir -p "${LOCAL_DIR}/slurm_logs"
  rsync -avz ${EXTRA_RSYNC_ARGS[@]+"${EXTRA_RSYNC_ARGS[@]}"} "${REMOTE_HOST}:${REMOTE_DIR}/slurm_logs/" "${LOCAL_DIR}/slurm_logs/"
else
  echo "==> Note: No remote slurm_logs/ found. Skipping."
fi

# 2. Run Telemetry & Logs
if remote_dir_exists "${REMOTE_DIR}/runs"; then
  echo "==> Syncing runs/ (logs and telemetry)..."
  mkdir -p "${LOCAL_DIR}/runs"
  rsync -avz ${EXTRA_RSYNC_ARGS[@]+"${EXTRA_RSYNC_ARGS[@]}"} "${REMOTE_HOST}:${REMOTE_DIR}/runs/" "${LOCAL_DIR}/runs/"
else
  echo "==> Note: No remote runs/ directory found. Skipping."
fi

# 3. Root Benchmark Output
if remote_file_exists "${REMOTE_DIR}/time_test_results.json"; then
  echo "==> Syncing time_test_results.json..."
  rsync -avz ${EXTRA_RSYNC_ARGS[@]+"${EXTRA_RSYNC_ARGS[@]}"} "${REMOTE_HOST}:${REMOTE_DIR}/time_test_results.json" "${LOCAL_DIR}/time_test_results.json"
fi

# 4. Models, Metrics & Evaluation Checkpoints
if remote_dir_exists "${REMOTE_DIR}/models"; then
  echo "==> Syncing models/ directory..."
  mkdir -p "${LOCAL_DIR}/models"

  if [ "$ALL_WEIGHTS" = true ]; then
    rsync -avz ${EXTRA_RSYNC_ARGS[@]+"${EXTRA_RSYNC_ARGS[@]}"} "${REMOTE_HOST}:${REMOTE_DIR}/models/" "${LOCAL_DIR}/models/"
  elif [ "$LOGS_ONLY" = true ]; then
    rsync -avz ${EXTRA_RSYNC_ARGS[@]+"${EXTRA_RSYNC_ARGS[@]}"} \
      --include='*/' \
      --include='*.csv' \
      --include='*.json' \
      --exclude='*.pt' \
      --exclude='*.pth' \
      --exclude='checkpoints/' \
      --exclude='*/checkpoints/' \
      "${REMOTE_HOST}:${REMOTE_DIR}/models/" "${LOCAL_DIR}/models/"
  else
    # Default: metrics, CSVs, JSONs, and best/final checkpoints only
    rsync -avz ${EXTRA_RSYNC_ARGS[@]+"${EXTRA_RSYNC_ARGS[@]}"} \
      --include='*/' \
      --include='*.csv' \
      --include='*.json' \
      --include='*_best.pt' \
      --include='*_final.pt' \
      --exclude='*.pt' \
      --exclude='*.pth' \
      --exclude='checkpoints/' \
      --exclude='*/checkpoints/' \
      "${REMOTE_HOST}:${REMOTE_DIR}/models/" "${LOCAL_DIR}/models/"
  fi
else
  echo "==> Note: No remote models/ directory found. Skipping."
fi

echo "==> Sync from HPC completed successfully."
