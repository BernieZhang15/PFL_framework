#!/usr/bin/env bash
#SBATCH -J FedFourierFT_no_bias
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
FLR=1
FR=1.0
FB=true
SEEDS=(
    $((RANDOM % 100000))
    $((RANDOM % 100000))
    $((RANDOM % 100000))
)
LOGDIR="./FedFourierFT_no_bias/slurm-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which $PY)"
echo "freq_ratio: ${FR}, freq_bias: ${FB}, seeds: ${SEEDS[*]}"

# Datasets to run
DATASETS=(
    "Cifar10-pat-2M"
    "Cifar10-pat-5M"
)

# Function to run experiment on specific GPU
run_experiment() {
    local gpu_id=$1
    local dataset=$2
    local seed=$3
    local logfile="${LOGDIR}/${dataset}_fr${FR}_fb${FB}_seed${seed}.log"
    
    echo "[GPU ${gpu_id}] Starting: dataset=${dataset}, fr=${FR}, fb=${FB}, seed=${seed}"
    $PY "$MAIN" -data "$dataset" -lr "$LR" -fr "$FR" -fb "$FB" -dlrr "$FLR" -seed "$seed" -did "$gpu_id" > "$logfile" 2>&1
    local exit_code=$?
    echo "[GPU ${gpu_id}] Finished: dataset=${dataset}, seed=${seed}, exit_code=${exit_code}"
    return $exit_code
}

# Total number of experiments
total_experiments=${#DATASETS[@]}
echo "Total experiments to run: ${total_experiments}"
echo "Using 3 GPUs in parallel"
echo "========================================"

# Run experiments in batches of 3 (one per GPU)
num_experiments=${#DATASETS[@]}
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "======== Running seed ${seed} ========"
    batch_idx=0

    while [ $batch_idx -lt $num_experiments ]; do
        echo ""
        echo "======== Batch starting at index ${batch_idx} ========"
        
        # Launch up to 3 experiments in parallel
        pids=()
        for gpu in 0 1 2; do
            exp_idx=$((batch_idx + gpu))
            if [ $exp_idx -lt $num_experiments ]; then
                ds="${DATASETS[$exp_idx]}"
                run_experiment $gpu "$ds" "$seed" &
                pids+=($!)
            fi
        done
        
        # Wait for all background jobs in this batch to complete
        for pid in "${pids[@]}"; do
            wait $pid || true
        done
        
        batch_idx=$((batch_idx + 3))
    done
done

echo ""
echo "========================================"
echo "All experiments finished at $(date)"
echo "Logs saved in: $LOGDIR"
