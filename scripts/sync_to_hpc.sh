#!/usr/bin/env bash
set -euo pipefail

# HPC cluster configuration (overrideable via environment variables)
REMOTE_HOST="${REMOTE_HOST:-"orfeo"}"
REMOTE_DIR="${REMOTE_DIR:-"/orfeo/scratch/dssc/kj_mersimoski/RL_project"}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Preparing remote directory on ${REMOTE_HOST}:${REMOTE_DIR}..."
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"

echo "==> Syncing local repository to HPC cluster (${REMOTE_HOST}:${REMOTE_DIR})..."
rsync -avz \
    --include='.env' \
    --include='uv.lock' \
    --filter=':- .gitignore' \
    --exclude='.git/' \
    --exclude='.agents/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='env/' \
    --exclude='.pytest_cache/' \
    --exclude='.uv_cache/' \
    --exclude='__pycache__/' \
    --exclude='docs/' \
    --exclude='tmp/' \
    --exclude='runs/' \
    --exclude='slurm_logs/' \
    --exclude='models/' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.h5' \
    --exclude='*.zip' \
    --exclude='*.safetensors' \
    --exclude='*.zarr' \
    "$@" \
    "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "==> Sync to HPC completed successfully."
