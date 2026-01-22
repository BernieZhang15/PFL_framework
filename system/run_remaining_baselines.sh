#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
MAIN=${MAIN:-main.py}
# Keep consistent with your Fourier runs by default; override via env if needed.
LR=0.01
LOGDIR=${LOGDIR:-./baselines}
mkdir -p "$LOGDIR"

DATASETS="
Cifar100-pat-5M
"

# Remaining experiments (from system/main.py algo dict):
# 2: ["pFB_cnn", "pFedBayes"]
# 3: ["cnn", "FedAvg"]
# 4: ["cnn", "FedProx"]
# 5: ["cnn", "Ditto"]
# 6: ["cnn", "PerAvg"]
# 7: ["cnn", "pFedMe"]
# 8: ["cnn", "FedRep"]
# EXPERIMENTS="
# pFB_cnn pFedBayes
# cnn FedAvg
# cnn FedProx
# cnn PerAvg
# cnn pFedMe
# "
EXPERIMENTS="
cnn FedAvg
"

run_one() {
  local model="$1"
  local algo="$2"
  local ds="$3"

  local tag="${algo}_${model}_${ds}"
  local logfile="$LOGDIR/${tag}.log"

  echo "============================================================"
  echo "Running: algo=${algo} model=${model} dataset=${ds} lr=${LR}"
  echo "Log: $logfile"

  "$PY" "$MAIN" -m "$model" -algo "$algo" -data "$ds" -lr "$LR" -nb 100 2>&1 | tee "$logfile"
}

echo "$EXPERIMENTS" | while read -r model algo; do
  [[ -z "${model:-}" ]] && continue
  echo "$DATASETS" | while IFS= read -r ds; do
    [[ -z "$ds" ]] && continue
    run_one "$model" "$algo" "$ds"
  done
done

echo "All runs finished. Logs in $LOGDIR"