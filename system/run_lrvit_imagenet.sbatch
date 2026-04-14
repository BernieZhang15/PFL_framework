#!/usr/bin/env bash
#SBATCH -J LRFedViT_imagenet
#SBATCH -p gpu-h100
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
LR=0.1
RANK=4
SEEDS=(576)
LOGDIR="./LRFedViT_imagenet/slurm-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

NUM_GPUS=1

echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which $PY)"
echo "LR: ${LR}, RANK: ${RANK}, Seeds: ${SEEDS[*]}"
echo "Number of GPUs: ${NUM_GPUS}"

DATASETS=(
    "Tiny-imagenet"
)

run_experiment() {
    local gpu_id=$1
    local dataset=$2
    local seed=$3
    local logfile="${LOGDIR}/${dataset}-seed${seed}.log"

    echo "[GPU ${gpu_id}] Starting: dataset=${dataset}, seed=${seed}"
    $PY "$MAIN" \
        -m "LRFedViT" \
        -algo "FedMetaBayes" \
        -data "$dataset" \
        -nb 200 \
        -nc 50 \
        -lr "$LR" \
        -rank "$RANK" \
        -seed "$seed" \
        -did "$gpu_id" \
        -gr 1000 \
        -ls 20 \
        -lbs 32 \
        > "$logfile" 2>&1
    local exit_code=$?
    echo "[GPU ${gpu_id}] Finished: dataset=${dataset}, seed=${seed}, exit_code=${exit_code}"
    return $exit_code
}

num_experiments=${#DATASETS[@]}
echo "Total experiments to run: ${num_experiments}"
echo "========================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "======== Running seed ${seed} ========"
    batch_idx=0

    while [ $batch_idx -lt $num_experiments ]; do
        echo ""
        echo "======== Batch starting at index ${batch_idx} ========"

        pids=()
        for gpu_id in 0; do
            exp_idx=$((batch_idx + gpu_id))
            if [ $exp_idx -lt $num_experiments ]; then
                dataset="${DATASETS[$exp_idx]}"
                run_experiment $gpu_id "$dataset" "$seed" &
                pids+=($!)
            fi
        done

        for pid in "${pids[@]}"; do
            wait $pid || true
        done

        batch_idx=$((batch_idx + NUM_GPUS))
    done
done

echo ""
echo "========================================"
echo "All experiments finished at $(date)"
echo "Logs saved in: $LOGDIR"
