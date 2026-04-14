#!/usr/bin/env bash
#SBATCH -J ViT_pFedMe_imagenet
#SBATCH -p gpu-l40s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o runs/slurm-%x-%j.out
#SBATCH -e runs/slurm-%x-%j.err

set -uo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs

# Load environment
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PFL

PY=${PY:-python}
MAIN=main.py
LR=0.01
SEEDS=(576)
LOGDIR="./FedViT_imagenet/slurm-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which $PY)"
echo "LR: ${LR}, Seeds: ${SEEDS[*]}"

DATASET="Tiny-imagenet"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "======== Running seed ${seed} ========"
    logfile="${LOGDIR}/${DATASET}-pFedMe-seed${seed}.log"
    echo "[Starting] algo=pFedMe, dataset=${DATASET}, seed=${seed}"

    $PY "$MAIN" \
        -m "fedvit" \
        -algo "pFedMe" \
        -data "$DATASET" \
        -nb 200 \
        -nc 50 \
        -lr "$LR" \
        -seed "$seed" \
        -did "0" \
        -gr 1000 \
        -ls 20 \
        -lbs 32 \
        -lam 3 \
        -lrp 0.01 \
        -K 10 \
        -bt 0.7 \
        -ldg 0.998 \
        > "$logfile" 2>&1
    exit_code=$?
    echo "[Finished] algo=pFedMe, seed=${seed}, exit_code=${exit_code}"
done

echo ""
echo "========================================"
echo "All experiments finished at $(date)"
