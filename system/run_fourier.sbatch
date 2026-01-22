#!/usr/bin/env bash
#SBATCH -J FedFourierFT
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
LR=0.1
FR=1.0
FLR=1
SEEDS=(
    $((RANDOM % 100000))
    $((RANDOM % 100000))
    $((RANDOM % 100000))
)
LOGDIR="./FedFourierFT/slurm-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

# Number of GPUs to use
NUM_GPUS=3

echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which $PY)"
echo "LR: ${LR}, FR: ${FR}, Seeds: ${SEEDS[*]}"
echo "Number of GPUs: ${NUM_GPUS}"

# Datasets to run
DATASETS=(
    "Cifar10-pat-2S"
    "Cifar10-pat-5S"
    "Cifar10-pat-2M"
    "Cifar10-pat-5M"
)

# Function to run experiment on specific GPU
run_experiment() {
    local gpu_id=$1
    local dataset=$2
    local seed=$3
    local logfile="${LOGDIR}/${dataset}-seed${seed}.log"
    
    echo "[GPU ${gpu_id}] Starting: dataset=${dataset}, seed=${seed}"
    $PY "$MAIN" -data "$dataset" -lr "$LR" -fr "$FR" -dlrr "$FLR" -seed "$seed" -did "$gpu_id" > "$logfile" 2>&1
    local exit_code=$?
    echo "[GPU ${gpu_id}] Finished: dataset=${dataset}, seed=${seed}, exit_code=${exit_code}"
    return $exit_code
}

# Total number of experiments
total_experiments=${#DATASETS[@]}
echo "Total experiments to run: ${total_experiments}"
echo "Using ${NUM_GPUS} GPUs in parallel"
echo "========================================"

# Run experiments in batches of NUM_GPUS (one per GPU)
num_experiments=${#DATASETS[@]}
batch_idx=0

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "======== Running seed ${seed} ========"
    batch_idx=0

    while [ $batch_idx -lt $num_experiments ]; do
        echo ""
        echo "======== Batch starting at index ${batch_idx} ========"

        # Launch up to NUM_GPUS experiments in parallel
        pids=()
        for gpu_id in 0 1 2; do
            exp_idx=$((batch_idx + gpu_id))
            if [ $exp_idx -lt $num_experiments ]; then
                dataset="${DATASETS[$exp_idx]}"
                run_experiment $gpu_id "$dataset" "$seed" &
                pids+=($!)
            fi
        done

        # Wait for all background jobs in this batch to complete
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
