#!/bin/bash
#SBATCH -A dssc
#SBATCH -p GPU
#SBATCH --job-name=order_chaos_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:V100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -Eeuo pipefail

SCRIPT_START_EPOCH=$(date +%s)

log() {
  local level="$1"
  shift
  printf '[%s] [%s] %s\n' "$(date -Is)" "$level" "$*"
}

# -------------------------------------------------------------
# Run Directories & Output Logging
# -------------------------------------------------------------
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="runs/${RUN_ID}"
mkdir -p slurm_logs "${LOG_DIR}" models

exec > >(tee -a "${LOG_DIR}/job.log") 2>&1

log "INFO" "=========================================================="
log "INFO" "   ORDER & CHAOS RL: V100 SLURM CURRICULUM TRAINING JOB    "
log "INFO" "=========================================================="
log "INFO" "Job ID        : ${SLURM_JOB_ID:-N/A (Interactive/Local)}"
log "INFO" "Node List     : ${SLURM_NODELIST:-localhost}"
log "INFO" "Run Directory : ${LOG_DIR}"

# -------------------------------------------------------------
# Background Resource Monitoring (NVIDIA-SMI)
# -------------------------------------------------------------
GPU_MONITORING_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  log "INFO" "Initializing NVIDIA-SMI GPU monitoring..."
  nvidia-smi | tee "${LOG_DIR}/nvidia-smi.start.txt" || true

  (
    echo "timestamp,index,memory.used(MiB),util.gpu(%),util.mem(%),temp(C),power(W)"
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
      --format=csv,noheader,nounits -l 1
  ) > "${LOG_DIR}/nvidia-smi.csv" 2>/dev/null &
  GPU_MONITORING_PID=$!
fi

cleanup() {
  local rc=$?
  if [[ -n "${GPU_MONITORING_PID}" ]] && kill -0 "${GPU_MONITORING_PID}" 2>/dev/null; then
    kill "${GPU_MONITORING_PID}" 2>/dev/null || true
    wait "${GPU_MONITORING_PID}" 2>/dev/null || true
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee "${LOG_DIR}/nvidia-smi.end.txt" || true

    if [[ -f "${LOG_DIR}/nvidia-smi.csv" ]] && [[ $(wc -l < "${LOG_DIR}/nvidia-smi.csv") -gt 1 ]]; then
      log "INFO" "=== Resource Utilization Summary ==="
      awk -F',' 'NR > 1 {
        if ($3 > max_mem) max_mem = $3;
        sum_util += $4;
        if ($4 > max_util) max_util = $4;
        sum_pwr += $7;
        count++;
      } END {
        if (count > 0) {
          printf "Peak GPU Memory: %d MiB\n", max_mem;
          printf "Avg GPU Util:    %.1f %%\n", sum_util / count;
          printf "Peak GPU Util:   %.1f %%\n", max_util;
          printf "Avg Power Draw:  %.1f W\n", sum_pwr / count;
        }
      }' "${LOG_DIR}/nvidia-smi.csv" | while IFS= read -r line; do log "INFO" "$line"; done
    fi
  fi

  local script_end_epoch
  script_end_epoch=$(date +%s)
  local elapsed=$((script_end_epoch - SCRIPT_START_EPOCH))
  if [ "$rc" -eq 0 ]; then
    log "INFO" "Training pipeline completed in ${elapsed}s ($((elapsed / 60)) min)."
  else
    log "ERROR" "Training pipeline failed after ${elapsed}s (exit code ${rc})."
  fi
}
trap cleanup EXIT INT TERM

# -------------------------------------------------------------
# Environment Setup & Curriculum Execution
# -------------------------------------------------------------
log "INFO" "Syncing Python virtual environment with uv..."
uv sync

log "INFO" "Launching 5-Stage Curriculum Training on V100 GPU (Double DQN)..."
uv run python src/run_curriculum.py \
  --device cuda \
  --double_dqn \
  --batch_size 256 \
  --buffer_capacity 100000 \
  --buffer_trim_size 20000 \
  --gamma 0.90 \
  --target_update_freq 1000 \
  --eval_interval 500 \
  --eval_episodes 40 \
  --holdout_episodes 100 \
  --output_dir models

log "INFO" "Training completed. All checkpoints and logs saved to models/."
