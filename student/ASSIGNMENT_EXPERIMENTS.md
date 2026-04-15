# Assignment 3 GPU Experiment Plan

The assignment handout asks for these GPU-facing experiment groups:

- `baseline`: zero-shot MATH baseline on 500 test examples.
- `sft_tune`: learning-rate tuning runs to get SFT loss decreasing substantially.
- `sft_sizes`: SFT runs for `128`, `256`, `512`, `1024`, and `full`.
- `grpo_lr`: Countdown GRPO learning-rate sweep with at least 3 learning rates.
- `grpo_baselines`: compare `reinforce_with_baseline` vs `no_baseline`.
- `grpo_length_norm`: compare `masked_mean` vs `masked_normalize`.
- `grpo_std_norm`: compare `use_std_normalization=true` vs `false`.

Use the registry to inspect commands:

```bash
python student/assignment_experiments.py groups
python student/assignment_experiments.py list --group sft_sizes
python student/assignment_experiments.py command --name sft-512
```

Submit a whole group on the cluster:

```bash
bash student/submit_assignment_group.sh baseline
bash student/submit_assignment_group.sh sft_tune
bash student/submit_assignment_group.sh sft_sizes
```

For the NYU HPC as `sv2279`, start by copying:

```bash
cp student/hpc_sv2279.env.example student/hpc_sv2279.env
```

Then fill in `WANDB_API_KEY` if you want online logging. The launcher automatically sources `student/hpc_sv2279.env` on the cluster.

The GRPO groups are already defined in the registry, but they are intentionally skipped by the submit script until `student/run_grpo_experiment.py` exists.

Each SFT run now writes:

- `results.json`
- `train_history.jsonl`
- `eval_history.jsonl`
- `train_loss_curve.png`
- `val_accuracy_curve.png`

Useful environment overrides:

- `A3_SAVE_ROOT`
- `A3_TRAIN_PATH`
- `A3_PRIME_VAL_PATH`
- `A3_PRIME_TEST_PATH`
- `A3_COUNTDOWN_TRAIN_PATH`
- `A3_COUNTDOWN_DEV_PATH`
- `A3_COUNTDOWN_TEST_PATH`
- `BEST_GRPO_LR`
- `BEST_GRPO_LOSS_TYPE`
- `BEST_LENGTH_NORM`
- `BEST_STD_NORM`
