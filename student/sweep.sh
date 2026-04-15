#!/bin/bash
set -euo pipefail

cd /scratch/sv2279/SFT-RL
mkdir -p logs

export SAVE_ROOT=/scratch/sv2279/sft-math-runs
mkdir -p $SAVE_ROOT

TRAIN_PATH=/scratch/sv2279/SFT-RL/data-distrib/intellect_math/train
PRIME_VAL_PATH=/scratch/sv2279/SFT-RL/data-distrib/intellect_math/dev
PRIME_TEST_PATH=/scratch/sv2279/SFT-RL/data-distrib/intellect_math/test

MODEL=Qwen/Qwen2.5-Math-1.5B
WANDB_PROJECT=sft-math

run_one () {
  NAME=$1
  LIMIT=${2:-}
  echo Starting run: $NAME
  COMMON_ARGS="--model $MODEL \
    --train-path $TRAIN_PATH \
    --prime-val-path $PRIME_VAL_PATH \
    --prime-test-path $PRIME_TEST_PATH \
    --output-dir $SAVE_ROOT/$NAME \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 1e-5 \
    --num-epochs 1 \
    --use-vllm \
    --policy-device cuda:0 \
    --eval-device cuda:1 \
    --gpu-memory-utilization 0.4 \
    --wandb-project $WANDB_PROJECT \
    --run-name $NAME"

  if [ -n "$LIMIT" ]; then
    uv run python student/run_sft_experiment.py \
      $COMMON_ARGS \
      --train-limit $LIMIT \
      --eval-every-steps 50
  else
    uv run python student/run_sft_experiment.py \
      $COMMON_ARGS \
      --eval-every-steps 100
  fi
  echo Finished run: $NAME
  cat $SAVE_ROOT/$NAME/results.json
}

run_one sft-128 128
run_one sft-256 256
run_one sft-512 512
run_one sft-1024 1024
run_one sft-full