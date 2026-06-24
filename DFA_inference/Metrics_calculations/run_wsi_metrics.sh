#!/bin/bash
#SBATCH --account=<acc name>
#SBATCH --partition=defq
#SBATCH --job-name=wsi_metrics
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/wsi_metrics_%j.out
#SBATCH --error=logs/wsi_metrics_%j.err

set -euo pipefail

echo "=========================================="
echo "WSI Metrics Calculation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_JOB_NODELIST:-N/A}"
echo "Started at: $(date '+%F %T')"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
eval "$(conda shell.bash hook)"
# Conda environment name
CONDA_ENV=${CONDA_ENV:-<your_conda_env_name>}

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "Python version:"
python3 --version
echo ""

echo "NumPy version:"
python3 -c "import numpy; print(numpy.__version__)"
echo ""

# Change to script directory
cd /path/to/script/directory

echo "Running WSI metrics calculation script..."
echo "=========================================="
echo ""

# Run the metrics calculation
python3 calculate_wsi_metrics.py

echo ""
echo "=========================================="
echo "Metrics calculation complete!"
echo "Finished at: $(date '+%F %T')"
echo "=========================================="