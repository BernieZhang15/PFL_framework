#!/usr/bin/env bash
#SBATCH -J FedFourierFT_sl_sweep_cifar100_5M
#SBATCH -p gpu-l40s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:3
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH -o runs/slurm-%x-%j.out
#SBATCH -e runs/slurm-%x-%j.err

# Don't use set -e as it causes issues with background processes
set -uo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs

# Load environment
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PFL

PY=${PY:-python}
MAIN=main.py
LR=0.001
FR=1.0
FLR=10
FB=true
SEEDS=(
    573
    592
    567
)
LOGDIR="./FedFourierFT_sl_sweep_cifar100_5M/slurm-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

# Number of GPUs to use (SLURM remaps GPUs to 0,1,2 inside the job)
NUM_GPUS=3

echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which $PY)"
echo "LR: ${LR}, FR: ${FR}, FB: ${FB}, Seeds: ${SEEDS[*]}"
echo "Number of GPUs: ${NUM_GPUS}"

# spec_lambda values: 0.1 ~ 1.0
SL_VALUES=(1 2 3 4 5 6 7 8 9 10)
# Datasets to run
DATASETS=(
    "Cifar100-pat-5M"
)

# Function to run experiment on specific GPU
run_experiment() {
    local gpu_id=$1
    local sl=$2
    local dataset=$3
    local seed=$4
    local logfile="${LOGDIR}/${dataset}_sl${sl}_seed${seed}.log"
    
    echo "[GPU ${gpu_id}] Starting: dataset=${dataset}, sl=${sl}, seed=${seed}"
    $PY "$MAIN" -data "$dataset" -lr "$LR" -fr "$FR" -fb "$FB" -dlrr "$FLR" -sl "$sl" -seed "$seed" -nb 100 -did "$gpu_id" > "$logfile" 2>&1
    local exit_code=$?
    echo "[GPU ${gpu_id}] Finished: dataset=${dataset}, sl=${sl}, seed=${seed}, exit_code=${exit_code}"
    return $exit_code
}

# Total number of experiments
total_experiments=$((${#DATASETS[@]} * ${#SL_VALUES[@]} * ${#SEEDS[@]}))
echo "Total experiments to run: ${total_experiments}"
echo "Using ${NUM_GPUS} GPUs in parallel"
echo "========================================"

# Create a list of all (dataset, sl, seed) combinations
declare -a EXPERIMENTS
idx=0
for ds in "${DATASETS[@]}"; do
    for sl in "${SL_VALUES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXPERIMENTS[$idx]="${ds}|${sl}|${seed}"
            idx=$((idx + 1))
        done
    done
done

# Run experiments in batches of NUM_GPUS (one per GPU)
num_experiments=${#EXPERIMENTS[@]}
batch_idx=0

while [ $batch_idx -lt $num_experiments ]; do
    echo ""
    echo "======== Batch starting at index ${batch_idx} ========"
    
    # Launch up to NUM_GPUS experiments in parallel
    pids=()
    for gpu_id in 0 1 2; do
        exp_idx=$((batch_idx + gpu_id))
        if [ $exp_idx -lt $num_experiments ]; then
            exp="${EXPERIMENTS[$exp_idx]}"
            ds="${exp%%|*}"
            rest="${exp#*|}"
            sl="${rest%%|*}"
            seed="${rest##*|}"
            run_experiment $gpu_id "$sl" "$ds" "$seed" &
            pids+=($!)
        fi
    done
    
    # Wait for all background jobs in this batch to complete
    for pid in "${pids[@]}"; do
        wait $pid || true
    done
    
    batch_idx=$((batch_idx + NUM_GPUS))
done

echo ""
echo "========================================"
echo "All experiments finished at $(date)"
echo "Logs saved in: $LOGDIR"
