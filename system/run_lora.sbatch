#!/usr/bin/env bash
#SBATCH -J FedLoRA
#SBATCH -p gpu-l40s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o runs/slurm-%x-%j.out
#SBATCH -e runs/slurm-%x-%j.err

set -euo pipefail

# Ensure relative paths (e.g. main.py, LOGDIR=./FedLoRA) resolve from the repo root
cd "${SLURM_SUBMIT_DIR}"

mkdir -p runs

# Optional: override log dir per job to avoid collisions
export LOGDIR="./FedLoRA/slurm-${SLURM_JOB_ID}"

# If you use conda/modules, uncomment and adjust:
# module load 
# source ~/miniconda3/etc/profile.d/conda.sh
module load miniforge3
conda activate PFL


bash ./lora.sh
