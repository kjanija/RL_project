#!/bin/bash
#SBATCH -A dssc
#SBATCH -p GPU
#SBATCH --job-name=time_test_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:V100:1
#SBATCH --time=00:15:00
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
LOG_DIR="runs/time_test_${RUN_ID}"
mkdir -p slurm_logs "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/time_test.log") 2>&1

log "INFO" "=========================================================="
log "INFO" "    ORDER & CHAOS RL: V100 TIME CALIBRATION BENCHMARK     "
log "INFO" "=========================================================="
log "INFO" "Job ID        : ${SLURM_JOB_ID:-N/A (Interactive/Local)}"
log "INFO" "Node List     : ${SLURM_NODELIST:-localhost}"
log "INFO" "Log Directory : ${LOG_DIR}"

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
    log "INFO" "Calibration benchmark finished in ${elapsed}s."
  else
    log "ERROR" "Calibration benchmark failed after ${elapsed}s (exit code ${rc})."
  fi
}
trap cleanup EXIT INT TERM

# -------------------------------------------------------------
# Environment Setup & Execution
# -------------------------------------------------------------
log "INFO" "Syncing Python virtual environment with uv..."
uv sync

log "INFO" "Running time_test.py (150 games on V100)..."
uv run python time_test.py --device cuda --batch_size 256 --test_episodes 150

# Copy benchmark results to the job log dir for archival
if [[ -f "time_test_results.json" ]]; then
  cp "time_test_results.json" "${LOG_DIR}/time_test_results.json"
fi

log "INFO" "Calibration completed. Results saved to time_test_results.json."
