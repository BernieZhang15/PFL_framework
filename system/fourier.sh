#!/bin/sh
set -e

PY=python
MAIN=main.py
LOGDIR=FedFourierFT
LR=0.15
mkdir -p "$LOGDIR"

DATASETS="
Cifar10-pat-5N
Cifar10-pat-2S
Cifar10-pat-5S
Cifar10-pat-2M
Cifar10-pat-5M
"

echo "$DATASETS" | while IFS= read -r ds; do
  [ -z "$ds" ] && continue
  echo "Running dataset: $ds"
  $PY "$MAIN" -data "$ds" -lr "$LR" > "$LOGDIR/${ds}.log" 2>&1
done

echo "All runs finished. Logs in $LOGDIR"