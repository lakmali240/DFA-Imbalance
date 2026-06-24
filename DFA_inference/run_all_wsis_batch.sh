#!/bin/bash
#SBATCH --account=<account_name>
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/batch_all_wsis_%j.out
#SBATCH --error=logs/batch_all_wsis_%j.err
#SBATCH --job-name=process_all_wsis

set -euo pipefail

# ============================================================================
# MASTER SCRIPT TO PROCESS ALL 10 WSIs
# ============================================================================

# ------------- Configuration -------------
WSI_CONFIG="./wsi_config.txt"
BASE_OUTPUT_DIR="/path/to/output"
SCRIPT_DIR="/home/DFA-Imbalance/DFA_inference"
RUNTIME_LOG="${BASE_OUTPUT_DIR}/runtime_summary.csv"

# SAMPath configuration
SAMPATH_DIR="/home/DFA-Imbalance"
SAMPATH_CONFIG="configs.BCSS_DFA" #configs.BDSA_DFA , configs.CRAG_DFA

SAMPATH_DFA_CHECKPOINT="./outputs/BCSS_DFA/<run_id>/checkpoints/last_best.ckpt"


# Processing parameters
PATCH_SIZE=1024
LEVEL=1
JPG_QUALITY=90
DATA_EXT=".jpg"

# ------------- Helper Functions -------------
hms() {
  local SECS=$1
  printf "%02d:%02d:%02d" $((SECS/3600)) $(((SECS%3600)/60)) $((SECS%60))
}

log_msg() {
  echo "[$(date '+%F %T')] $*"
}

# ------------- Setup -------------
mkdir -p logs
mkdir -p "${BASE_OUTPUT_DIR}"

log_msg "=========================================="
log_msg "BATCH PROCESSING ALL WSIs"
log_msg "=========================================="
log_msg "Job ID: ${SLURM_JOB_ID:-N/A}"
log_msg "Node: ${SLURM_JOB_NODELIST:-N/A}"
log_msg "Config file: ${WSI_CONFIG}"

nvidia-smi || true

# Activate conda environment
eval "$(conda shell.bash hook)"
# Conda environment name
CONDA_ENV=${CONDA_ENV:-<your_conda_env_name>}

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Initialize runtime log
echo "WSI_ID,WSI_Name,Extraction_Time_s,Prediction_Time_s,Stitching_Time_s,Visualization_Time_s,Total_Time_s,Status" > "${RUNTIME_LOG}"

# ------------- Process Each WSI -------------
BATCH_START=$(date +%s)
TOTAL_WSIS=0
SUCCESS_COUNT=0
FAILED_COUNT=0

while IFS='|' read -r WSI_ID WSI_PATH || [ -n "$WSI_ID" ]; do
  # Skip comments and empty lines
  [[ "$WSI_ID" =~ ^#.*$ ]] && continue
  [[ -z "$WSI_ID" ]] && continue
  
  TOTAL_WSIS=$((TOTAL_WSIS + 1))
  WSI_START=$(date +%s)
  
  # Extract WSI name from path
  WSI_NAME=$(basename "${WSI_PATH}" .png)
  
  log_msg ""
  log_msg "=========================================="
  log_msg "Processing WSI ${WSI_ID}: ${WSI_NAME}"
  log_msg "=========================================="
  log_msg "Path: ${WSI_PATH}"
  
  # Check if WSI exists
  if [ ! -f "${WSI_PATH}" ]; then
    log_msg "ERROR: WSI file not found: ${WSI_PATH}"
    echo "${WSI_ID},${WSI_NAME},0,0,0,0,0,FILE_NOT_FOUND" >> "${RUNTIME_LOG}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi
  
  # Create output directories for this WSI
  WSI_OUTPUT_DIR="${BASE_OUTPUT_DIR}/WSI_${WSI_ID}_${WSI_NAME}"
  PATCHES_DIR="${WSI_OUTPUT_DIR}/original_patches"
  PREDICTIONS_DIR="${WSI_OUTPUT_DIR}/predicted_patches"
  RESULTS_DIR="${WSI_OUTPUT_DIR}/results"
  
  mkdir -p "${PATCHES_DIR}"
  mkdir -p "${PREDICTIONS_DIR}"
  mkdir -p "${RESULTS_DIR}"
  
  # Initialize timing variables
  EXTRACT_TIME=0
  PREDICT_TIME=0
  STITCH_TIME=0
  VIZ_TIME=0
  STATUS="SUCCESS"
  
  # # ========== STEP 1: Patch Extraction ==========
  log_msg "STEP 1: Extracting patches..."
  STEP_START=$(date +%s)
  
  if python3 "${SCRIPT_DIR}/patch_extraction.py" \
    --wsi_path "${WSI_PATH}" \
    --save_dir "${PATCHES_DIR}" \
    --patch_size ${PATCH_SIZE} \
    --level ${LEVEL} \
    --jpg_quality ${JPG_QUALITY}; then
    
    STEP_END=$(date +%s)
    EXTRACT_TIME=$((STEP_END - STEP_START))
    log_msg "✓ Extraction completed in $(hms ${EXTRACT_TIME})"
  else
    log_msg "✗ Extraction FAILED"
    STATUS="EXTRACTION_FAILED"
    echo "${WSI_ID},${WSI_NAME},${EXTRACT_TIME},0,0,0,${EXTRACT_TIME},${STATUS}" >> "${RUNTIME_LOG}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi
  
  # ========== STEP 2: SAMPath Prediction ==========
  log_msg "STEP 2: Running SAMPath DFA prediction..."
  STEP_START=$(date +%s)
  
  # Clean prediction directory
  rm -rf "${PREDICTIONS_DIR}"/*
  
  if python3 "${SAMPATH_DIR}/predict.py" \
    --config ${SAMPATH_CONFIG} \
    --input_dir "${PATCHES_DIR}/patches" \
    --data_ext ${DATA_EXT} \
    --output_dir "${PREDICTIONS_DIR}" \
    --pretrained "${SAMPATH_DFA_CHECKPOINT}" \
    --devices 0; then
    
    STEP_END=$(date +%s)
    PREDICT_TIME=$((STEP_END - STEP_START))
    log_msg "✓ Prediction completed in $(hms ${PREDICT_TIME})"
  else
    log_msg "✗ Prediction FAILED"
    STATUS="PREDICTION_FAILED"
    TOTAL_TIME=$((EXTRACT_TIME + PREDICT_TIME))
    echo "${WSI_ID},${WSI_NAME},${EXTRACT_TIME},${PREDICT_TIME},0,0,${TOTAL_TIME},${STATUS}" >> "${RUNTIME_LOG}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi
  
  # ========== STEP 3: Stitch WSI ==========
  log_msg "STEP 3: Stitching patches back to WSI..."
  STEP_START=$(date +%s)
  
  if python3 "${SCRIPT_DIR}/WSI_patch_stitching.py" \
    --original_wsi_path "${WSI_PATH}" \
    --patches_folder "${PREDICTIONS_DIR}" \
    --output_path "${RESULTS_DIR}/stitched_wsi_mask.png" \
    --patch_size ${PATCH_SIZE} \
    --level ${LEVEL}; then
    
    STEP_END=$(date +%s)
    STITCH_TIME=$((STEP_END - STEP_START))
    log_msg "✓ Stitching completed in $(hms ${STITCH_TIME})"
  else
    log_msg "✗ Stitching FAILED"
    STATUS="STITCHING_FAILED"
    TOTAL_TIME=$((EXTRACT_TIME + PREDICT_TIME + STITCH_TIME))
    echo "${WSI_ID},${WSI_NAME},${EXTRACT_TIME},${PREDICT_TIME},${STITCH_TIME},0,${TOTAL_TIME},${STATUS}" >> "${RUNTIME_LOG}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi
  
  # ========== STEP 4: Visualization ==========
  log_msg "STEP 4: Creating visualizations..."
  STEP_START=$(date +%s)
  
  if python3 "${SCRIPT_DIR}/visualization_stitched_WSI.py" \
    --wsi_mask_path "${RESULTS_DIR}/stitched_wsi_mask.png" \
    --colored_save_path "${RESULTS_DIR}/colored_wsi.png" \
    --grayscale_save_path "${RESULTS_DIR}/grayscale_wsi.png" \
    --output_size 1024 1024; then
    
    STEP_END=$(date +%s)
    VIZ_TIME=$((STEP_END - STEP_START))
    log_msg "✓ Visualization completed in $(hms ${VIZ_TIME})"
  else
    log_msg "✗ Visualization FAILED"
    STATUS="VISUALIZATION_FAILED"
  fi
  
  # Calculate total time for this WSI
  WSI_END=$(date +%s)
  WSI_TOTAL=$((WSI_END - WSI_START))
  
  # Log runtime
  echo "${WSI_ID},${WSI_NAME},${EXTRACT_TIME},${PREDICT_TIME},${STITCH_TIME},${VIZ_TIME},${WSI_TOTAL},${STATUS}" >> "${RUNTIME_LOG}"
  
  log_msg "=========================================="
  log_msg "WSI ${WSI_ID} completed: ${STATUS}"
  log_msg "Total time: $(hms ${WSI_TOTAL})"
  log_msg "=========================================="
  
  if [ "${STATUS}" == "SUCCESS" ]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    FAILED_COUNT=$((FAILED_COUNT + 1))
  fi
  
  # Clean up intermediate patches to save space (optional)
  # Uncomment if disk space is limited:
  # rm -rf "${PATCHES_DIR}/patches"
  # rm -rf "${PREDICTIONS_DIR}"
  
done < "${WSI_CONFIG}"

# ------------- Final Summary -------------
BATCH_END=$(date +%s)
BATCH_TOTAL=$((BATCH_END - BATCH_START))

log_msg ""
log_msg "=========================================="
log_msg "BATCH PROCESSING COMPLETE"
log_msg "=========================================="
log_msg "Total WSIs processed: ${TOTAL_WSIS}"
log_msg "Successful: ${SUCCESS_COUNT}"
log_msg "Failed: ${FAILED_COUNT}"
log_msg "Total batch time: $(hms ${BATCH_TOTAL})"
log_msg "Runtime log: ${RUNTIME_LOG}"
log_msg "=========================================="

# Generate summary report
python3 "${SCRIPT_DIR}/generate_runtime_report.py" --runtime_log "${RUNTIME_LOG}"

log_msg "All done!"