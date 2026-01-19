#!/usr/bin/env bash
#SBATCH -J FedFourierFT_sweep
#SBATCH -p gpu-l40s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:3
#SBATCH -t 72:00:00
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
LR=0.15
SEED=42
LOGDIR="./FedFourierFT_sweep/slurm-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

# Number of GPUs to use (SLURM remaps GPUs to 0,1,2 inside the job)
NUM_GPUS=3

echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which $PY)"
echo "Seed: ${SEED}"
echo "Number of GPUs: ${NUM_GPUS}"

# freq_ratio values: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
FR_VALUES=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)

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
    local fr=$2
    local dataset=$3
    local logfile="${LOGDIR}/${dataset}_fr${fr}.log"
    
    echo "[GPU ${gpu_id}] Starting: dataset=${dataset}, fr=${fr}"
    $PY "$MAIN" -data "$dataset" -lr "$LR" -fr "$fr" -seed "$SEED" -did "$gpu_id" > "$logfile" 2>&1
    local exit_code=$?
    echo "[GPU ${gpu_id}] Finished: dataset=${dataset}, fr=${fr}, exit_code=${exit_code}"
    return $exit_code
}

# Total number of experiments
total_experiments=$((${#DATASETS[@]} * ${#FR_VALUES[@]}))
echo "Total experiments to run: ${total_experiments}"
echo "Using 3 GPUs in parallel"
echo "========================================"

# Create a list of all (dataset, fr) combinations
declare -a EXPERIMENTS
idx=0
for ds in "${DATASETS[@]}"; do
    for fr in "${FR_VALUES[@]}"; do
        EXPERIMENTS[$idx]="${ds}|${fr}"
        idx=$((idx + 1))
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
            fr="${exp##*|}"
            run_experiment $gpu_id "$fr" "$ds" &
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
