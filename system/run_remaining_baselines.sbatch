#!/usr/bin/env bash
#SBATCH -J FedBaselines
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

# Run from the submission directory (more reliable than BASH_SOURCE under Slurm).
# If you submit from repo root, this will cd into ./system automatically.
cd "${SLURM_SUBMIT_DIR}"
if [[ -f "run_remaining_baselines.sh" ]]; then
  :
elif [[ -f "system/run_remaining_baselines.sh" ]]; then
  cd system
else
  echo "ERROR: Cannot find run_remaining_baselines.sh from ${SLURM_SUBMIT_DIR}" >&2
  exit 2
fi

# Best-effort (Slurm may already have created the output files).
mkdir -p runs || true

# Put per-job logs under a unique directory to avoid collisions.
export LOGDIR="./baselines/slurm-${SLURM_JOB_ID}"

# If you use conda/modules, adjust these lines for your cluster.
if command -v module >/dev/null 2>&1; then
  module load miniforge3
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate PFL
else
  echo "ERROR: conda not found. Load your environment (module/conda) before running." >&2
  exit 127
fi

bash ./run_remaining_baselines.sh
