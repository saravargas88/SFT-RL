from __future__ import annotations

import argparse
import os
import shlex
from dataclasses import dataclass
from pathlib import Path


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    description: str
    command: list[str]
    runner_path: str

    @property
    def runner_exists(self) -> bool:
        return Path(self.runner_path).exists()


def build_experiments() -> list[Experiment]:
    model_base = env("A3_MODEL_BASE", "Qwen/Qwen2.5-Math-1.5B")
    model_instruct = env("A3_MODEL_INSTRUCT", "Qwen/Qwen2.5-Math-1.5B-Instruct")

    train_path = env("A3_TRAIN_PATH", "/scratch/sv2279/SFT-RL/data-distrib/intellect_math/train")
    prime_val_path = env("A3_PRIME_VAL_PATH", "/scratch/sv2279/SFT-RL/data-distrib/intellect_math/dev")
    prime_test_path = env("A3_PRIME_TEST_PATH", "/scratch/sv2279/SFT-RL/data-distrib/intellect_math/test")
    countdown_train_path = env("A3_COUNTDOWN_TRAIN_PATH", "/scratch/sv2279/SFT-RL/data-distrib/countdown/train")
    countdown_dev_path = env("A3_COUNTDOWN_DEV_PATH", "/scratch/sv2279/SFT-RL/data-distrib/countdown/dev")
    countdown_test_path = env("A3_COUNTDOWN_TEST_PATH", "/scratch/sv2279/SFT-RL/data-distrib/countdown/test")
    save_root = env("A3_SAVE_ROOT", "/scratch/sv2279/assignment3-runs")

    best_grpo_lr = env("BEST_GRPO_LR", "1e-5")
    best_grpo_loss_type = env("BEST_GRPO_LOSS_TYPE", "reinforce_with_baseline")
    best_length_norm = env("BEST_LENGTH_NORM", "masked_mean")
    best_std_norm = env("BEST_STD_NORM", "true")

    sft_common = [
        "uv",
        "run",
        "python",
        "student/run_sft_experiment.py",
        "--model",
        model_base,
        "--train-path",
        train_path,
        "--prime-val-path",
        prime_val_path,
        "--prime-test-path",
        prime_test_path,
        "--per-device-batch-size",
        env("A3_SFT_PER_DEVICE_BATCH_SIZE", "1"),
        "--gradient-accumulation-steps",
        env("A3_SFT_GRAD_ACCUM", "16"),
        "--learning-rate",
        env("A3_SFT_LR", "1e-5"),
        "--num-epochs",
        env("A3_SFT_EPOCHS", "1"),
        "--eval-every-steps",
        env("A3_SFT_EVAL_EVERY", "50"),
        "--policy-device",
        env("A3_POLICY_DEVICE", "cuda:0"),
        "--eval-device",
        env("A3_EVAL_DEVICE", "cuda:1"),
        "--wandb-project",
        env("A3_SFT_WANDB_PROJECT", "sft-math"),
        "--run-name",
    ]

    grpo_common = [
        "uv",
        "run",
        "python",
        "student/run_grpo_experiment.py",
        "--model",
        model_instruct,
        "--train-path",
        countdown_train_path,
        "--val-path",
        countdown_dev_path,
        "--test-path",
        countdown_test_path,
        "--prompt-name",
        "countdown",
        "--n-grpo-steps",
        "200",
        "--rollout-batch-size",
        "16",
        "--group-size",
        "8",
        "--sampling-temperature",
        "0.7",
        "--sampling-min-tokens",
        "4",
        "--sampling-max-tokens",
        "1024",
        "--epochs-per-rollout-batch",
        "1",
        "--train-batch-size",
        "64",
        "--gradient-accumulation-steps",
        "128",
        "--gpu-memory-utilization",
        "0.8",
        "--policy-device",
        env("A3_POLICY_DEVICE", "cuda:0"),
        "--eval-device",
        env("A3_EVAL_DEVICE", "cuda:1"),
        "--wandb-project",
        env("A3_GRPO_WANDB_PROJECT", "grpo-countdown"),
        "--run-name",
    ]

    exps: list[Experiment] = [
        Experiment(
            name="baseline-math-zero-shot",
            group="baseline",
            description="Zero-shot Qwen2.5-Math-1.5B baseline on 500 MATH test examples.",
            runner_path="student/evaluate_assignment_baseline.py",
            command=[
                "uv",
                "run",
                "python",
                "student/evaluate_assignment_baseline.py",
                "--model",
                model_base,
                "--dataset",
                "math",
                "--math-split",
                "test",
                "--max-examples",
                "500",
                "--prompt-name",
                "intellect",
                "--output-dir",
                f"{save_root}/baseline-math-zero-shot",
            ],
        ),
    ]

    for lr in ("5e-6", "1e-5", "2e-5"):
        name = f"sft-tune-lr-{lr.replace('-', '').replace('.', 'p')}"
        exps.append(
            Experiment(
                name=name,
                group="sft_tune",
                description=f"SFT tuning run on 1024 examples with learning rate {lr}.",
                runner_path="student/run_sft_experiment.py",
                command=sft_common
                + [
                    name,
                    "--output-dir",
                    f"{save_root}/{name}",
                    "--train-limit",
                    "1024",
                    "--learning-rate",
                    lr,
                ],
            )
        )

    for size in ("128", "256", "512", "1024"):
        name = f"sft-{size}"
        exps.append(
            Experiment(
                name=name,
                group="sft_sizes",
                description=f"SFT run on {size} Prime Intellect examples.",
                runner_path="student/run_sft_experiment.py",
                command=sft_common
                + [
                    name,
                    "--output-dir",
                    f"{save_root}/{name}",
                    "--train-limit",
                    size,
                ],
            )
        )

    exps.append(
        Experiment(
            name="sft-full",
            group="sft_sizes",
            description="SFT run on the full Prime Intellect training set.",
            runner_path="student/run_sft_experiment.py",
            command=sft_common
            + [
                "sft-full",
                "--output-dir",
                f"{save_root}/sft-full",
            ],
        )
    )

    for lr in ("5e-6", "1e-5", "2e-5"):
        name = f"grpo-lr-{lr.replace('-', '').replace('.', 'p')}"
        exps.append(
            Experiment(
                name=name,
                group="grpo_lr",
                description=f"Countdown GRPO run for learning-rate sweep at {lr}.",
                runner_path="student/run_grpo_experiment.py",
                command=grpo_common
                + [
                    name,
                    "--output-dir",
                    f"{save_root}/{name}",
                    "--learning-rate",
                    lr,
                    "--loss-type",
                    "reinforce_with_baseline",
                    "--use-std-normalization",
                    "true",
                    "--length-normalization",
                    "masked_mean",
                ],
            )
        )

    for loss_type in ("reinforce_with_baseline", "no_baseline"):
        name = f"grpo-baseline-{loss_type}"
        exps.append(
            Experiment(
                name=name,
                group="grpo_baselines",
                description=f"Countdown GRPO baseline comparison with {loss_type}.",
                runner_path="student/run_grpo_experiment.py",
                command=grpo_common
                + [
                    name,
                    "--output-dir",
                    f"{save_root}/{name}",
                    "--learning-rate",
                    best_grpo_lr,
                    "--loss-type",
                    loss_type,
                    "--use-std-normalization",
                    "true",
                    "--length-normalization",
                    "masked_mean",
                ],
            )
        )

    for length_norm in ("masked_mean", "masked_normalize"):
        name = f"grpo-length-{length_norm}"
        exps.append(
            Experiment(
                name=name,
                group="grpo_length_norm",
                description=f"Countdown GRPO length-normalization comparison with {length_norm}.",
                runner_path="student/run_grpo_experiment.py",
                command=grpo_common
                + [
                    name,
                    "--output-dir",
                    f"{save_root}/{name}",
                    "--learning-rate",
                    best_grpo_lr,
                    "--loss-type",
                    best_grpo_loss_type,
                    "--use-std-normalization",
                    best_std_norm,
                    "--length-normalization",
                    length_norm,
                ],
            )
        )

    for use_std in ("true", "false"):
        name = f"grpo-stdnorm-{use_std}"
        exps.append(
            Experiment(
                name=name,
                group="grpo_std_norm",
                description=f"Countdown GRPO group std-normalization comparison with use_std_normalization={use_std}.",
                runner_path="student/run_grpo_experiment.py",
                command=grpo_common
                + [
                    name,
                    "--output-dir",
                    f"{save_root}/{name}",
                    "--learning-rate",
                    best_grpo_lr,
                    "--loss-type",
                    best_grpo_loss_type,
                    "--use-std-normalization",
                    use_std,
                    "--length-normalization",
                    best_length_norm,
                ],
            )
        )

    return exps


def get_experiment(name: str) -> Experiment:
    for experiment in build_experiments():
        if experiment.name == name:
            return experiment
    raise SystemExit(f"Unknown experiment: {name}")


def list_experiments(group: str | None) -> None:
    for experiment in build_experiments():
        if group is not None and experiment.group != group:
            continue
        readiness = "ready" if experiment.runner_exists else "missing-runner"
        print(f"{experiment.name}\t{experiment.group}\t{readiness}\t{experiment.description}")


def list_groups() -> None:
    groups = sorted({experiment.group for experiment in build_experiments()})
    for group in groups:
        print(group)


def print_command(name: str, allow_missing_runner: bool) -> None:
    experiment = get_experiment(name)
    if not allow_missing_runner and not experiment.runner_exists:
        raise SystemExit(
            f"{experiment.name} expects {experiment.runner_path}, but that runner does not exist yet."
        )
    print(shlex.join(experiment.command))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment experiment registry and command generator.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--group", default=None)

    subparsers.add_parser("groups")

    command_parser = subparsers.add_parser("command")
    command_parser.add_argument("--name", required=True)
    command_parser.add_argument("--allow-missing-runner", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subcommand == "list":
        list_experiments(args.group)
        return
    if args.subcommand == "groups":
        list_groups()
        return
    if args.subcommand == "command":
        print_command(args.name, args.allow_missing_runner)
        return
    raise SystemExit(f"Unsupported subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
