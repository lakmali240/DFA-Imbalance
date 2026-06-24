#!/bin/bash
#SBATCH --account=<acc name>
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_adaptive_focal_attention.out
#SBATCH --error=logs/%j_adaptive_focal_attention.err


# ============================================================
# DFA : SAMPath with Adaptive Focal Attention Training Script
# ============================================================
# This script trains SAMPath with LEARNABLE focal attention that
# adapts based on actual class performance during training.
# ============================================================

eval "$(conda shell.bash hook)"
# Activate conda environment
CONDA_ENV=${CONDA_ENV:-<your_conda_env_name>}

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

nvidia-smi

mkdir -p logs


cd /home/lra240/Innov_SAMPATH/DFA-Imbalance

python main_save_best.py \
    --config configs.BDSA_DFA \
    --devices 0 \
    --project DFA_BDSA \
    --name BDSA_warmstart

# python main_save_best.py \
#     --config configs.BCSS_DFA \
#     --devices 0 \
#     --project DFA_BCSS \
#     --name BCSS_warmstart

echo "Training completed!"
