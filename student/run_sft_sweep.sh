#!/bin/bash
#SBATCH --job-name=sft_sweep
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --requeue

singularity exec --bind /scratch --nv \
--overlay /scratch/sv2279/overlay-25GB-500K.ext3:r \
/scratch/sv2279/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
set -euo pipefail

export WANDB_API_KEY=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

conda activate llmr
cd /scratch/sv2279/SFT-RL

mkdir -p logs
export SAVE_ROOT=/scratch/sv2279/sft-math-runs
mkdir -p \$SAVE_ROOT

TRAIN_PATH=/scratch/sv2279/SFT-RL/data-distrib/intellect_math/train
PRIME_VAL_PATH=/scratch/sv2279/SFT-RL/data-distrib/intellect_math/dev
PRIME_TEST_PATH=/scratch/sv2279/SFT-RL/data-distrib/intellect_math/test

MODEL=Qwen/Qwen2.5-Math-1.5B
WANDB_PROJECT=sft-math

run_one () {
  NAME=\$1
  LIMIT=\${2:-}

  echo Starting run: \$NAME

  if [ -n \"\$LIMIT\" ]; then
    uv run python student/run_sft_experiment.py \
      --model \$MODEL \
      --train-path \$TRAIN_PATH \
      --prime-val-path \$PRIME_VAL_PATH \
      --prime-test-path \$PRIME_TEST_PATH \
      --output-dir \$SAVE_ROOT/\$NAME \
      --train-limit \$LIMIT \
      --per-device-batch-size 1 \
      --gradient-accumulation-steps 16 \
      --learning-rate 1e-5 \
      --num-epochs 1 \
      --eval-every-steps 50 \
      --use-vllm \
      --policy-device cuda:0 \
      --eval-device cuda:1 \
      --wandb-project \$WANDB_PROJECT \
      --run-name \$NAME
  else
    uv run python student/run_sft_experiment.py \
      --model \$MODEL \
      --train-path \$TRAIN_PATH \
      --prime-val-path \$PRIME_VAL_PATH \
      --prime-test-path \$PRIME_TEST_PATH \
      --output-dir \$SAVE_ROOT/\$NAME \
      --per-device-batch-size 1 \
      --gradient-accumulation-steps 16 \
      --learning-rate 1e-5 \
      --num-epochs 1 \
      --eval-every-steps 100 \
      --use-vllm \
      --policy-device cuda:0 \
      --eval-device cuda:1 \
      --wandb-project \$WANDB_PROJECT \
      --run-name \$NAME
  fi

  echo Finished run: \$NAME
  cat \$SAVE_ROOT/\$NAME/results.json
}

run_one sft-128 128
run_one sft-256 256
run_one sft-512 512
run_one sft-1024 1024
run_one sft-full
"
