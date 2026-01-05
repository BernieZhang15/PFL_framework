#!/bin/sh
set -e

PY=python
MAIN=main.py
MODEL=lr_bayes_cnn
ALGO=FedMetaBayes
LR=0.001
LS=1
LOGDIR=logs_meta_bayes
mkdir -p "$LOGDIR"

DATASETS="
Cifar10-pat-5N
Cifar10-pat-2S
Cifar10-pat-5S
Cifar10-pat-2M
Cifar10-pat-5M"

echo "$DATASETS" | while IFS= read -r ds; do
  [ -z "$ds" ] && continue
  echo "Running dataset: $ds"
  $PY "$MAIN" -m "$MODEL" -algo "$ALGO" -data "$ds" -lr "$LR" -ls "$LS" > "$LOGDIR/${ds}.log" 2>&1
done

echo "All runs finished. Logs in $LOGDIR"