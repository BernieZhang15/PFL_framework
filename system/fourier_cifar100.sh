#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
MAIN=main.py
LOGDIR=${LOGDIR:-./FedFourierFT}
LR=0.001
NB=100
mkdir -p "$LOGDIR"

DATASETS="
Cifar100-pat-5M
"

echo "$DATASETS" | while IFS= read -r ds; do
  [ -z "$ds" ] && continue
  echo "Running dataset: $ds"
  $PY "$MAIN" -m "Fourier_bayes_cnn" -data "$ds" -lr "$LR" -fr 1 -sl 0.9 -nb "$NB" 2>&1 | tee "$LOGDIR/${ds}.log"
done

echo "All runs finished. Logs in $LOGDIR"