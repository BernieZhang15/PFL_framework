#!/usr/bin/env bash
#SBATCH -J FedFourierFT_cifar100_no_bias
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
FLR=10
FR=1.0
FB=true
SEEDS=(
    $((RANDOM % 100000))
    $((RANDOM % 100000))
    $((RANDOM % 100000))
)
LOGDIR="./FedFourierFT_cifar100_no_bias/slurm-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

# Number of GPUs to use
NUM_GPUS=3

echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which $PY)"
echo "LR: ${LR}, FR: ${FR}, FLR: ${FLR}, FB: ${FB}, Seeds: ${SEEDS[*]}"
echo "Number of GPUs: ${NUM_GPUS}"

# Datasets to run
DATASETS=(
    "Cifar100-pat-5M"
)

# Function to run experiment on specific GPU
run_experiment() {
    local gpu_id=$1
    local dataset=$2
    local seed=$3
    local logfile="${LOGDIR}/${dataset}_fr${FR}_fb${FB}_seed${seed}.log"
    
    echo "[GPU ${gpu_id}] Starting: dataset=${dataset}, fr=${FR}, fb=${FB}, seed=${seed}"
    $PY "$MAIN" -data "$dataset" -lr "$LR" -fr "$FR" -fb "$FB" -dlrr "$FLR" -seed "$seed" -nb 100 -did "$gpu_id" > "$logfile" 2>&1
    local exit_code=$?
    echo "[GPU ${gpu_id}] Finished: dataset=${dataset}, seed=${seed}, exit_code=${exit_code}"
    return $exit_code
}

# Total number of experiments
total_experiments=$((${#DATASETS[@]} * ${#SEEDS[@]}))
echo "Total experiments to run: ${total_experiments}"
echo "Using ${NUM_GPUS} GPUs in parallel"
echo "========================================"

dataset="${DATASETS[0]}"
num_experiments=${#SEEDS[@]}
batch_idx=0

while [ $batch_idx -lt $num_experiments ]; do
    echo ""
    echo "======== Batch starting at index ${batch_idx} ========"

    # Launch up to NUM_GPUS experiments in parallel
    pids=()
    for gpu_id in 0 1 2; do
        seed_idx=$((batch_idx + gpu_id))
        if [ $seed_idx -lt $num_experiments ]; then
            seed="${SEEDS[$seed_idx]}"
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

echo ""
echo "========================================"
echo "All experiments finished at $(date)"
echo "Logs saved in: $LOGDIR"
