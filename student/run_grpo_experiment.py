from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import wandb
from datasets import load_from_disk
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.grpo import (
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    masked_mean,
)
from student.sft import get_response_log_probs, masked_normalize, tokenize_prompt_and_output


# ── Countdown reward ──────────────────────────────────────────────────────────

def _extract_answer_text(response: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # generation may have been cut before closing tag
    m = re.search(r"<answer>(.*)", response, re.DOTALL)
    return m.group(1).strip() if m else None


def _safe_eval(expr: str) -> float | None:
    """Evaluate a plain arithmetic expression with no builtins."""
    cleaned = re.sub(r"[^0-9+\-*/().\s]", "", expr)
    if not cleaned.strip():
        return None
    try:
        return float(eval(cleaned, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _steps_reach_target(answer_text: str, target: int) -> bool:
    for line in reversed(answer_text.split("\n")):
        m = re.search(r"=\s*([-+]?\d+(?:\.\d+)?)\s*$", line.strip())
        if m:
            try:
                if abs(float(m.group(1)) - target) < 1e-6:
                    return True
            except ValueError:
                pass
    return False


def countdown_reward_fn(response: str, ground_truth: Any) -> dict[str, float]:
    """Binary reward for the Countdown task."""
    # parse target
    try:
        if isinstance(ground_truth, dict):
            target = int(ground_truth["target"])
        elif isinstance(ground_truth, (int, float)):
            target = int(ground_truth)
        else:
            target = int(str(ground_truth).strip())
    except (ValueError, KeyError, TypeError):
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    if "<answer>" not in response or "</answer>" not in response:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    answer_text = _extract_answer_text(response)
    if not answer_text:
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}

    # step-by-step check
    if _steps_reach_target(answer_text, target):
        return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}

    # single-expression check
    result = _safe_eval(answer_text)
    if result is not None and abs(result - target) < 1e-6:
        return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}

    # try each line
    for line in answer_text.split("\n"):
        line = re.sub(r"^Step\s+\d+:\s*", "", line.strip())
        if "=" in line:
            for part in line.split("="):
                r = _safe_eval(part.strip())
                if r is not None and abs(r - target) < 1e-6:
                    return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
        else:
            r = _safe_eval(line)
            if r is not None and abs(r - target) < 1e-6:
                return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}

    return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_prompt(name: str) -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text().strip()


def extract_countdown_example(
    example: dict[str, Any], prompt_template: str
) -> tuple[str, Any]:
    """Return (prompt, ground_truth) from a Countdown dataset row."""
    question = (
        example.get("question")
        or example.get("problem")
        or ""
    )
    if not question:
        nums = example.get("nums") or example.get("numbers") or []
        target = example.get("target", 0)
        question = (
            f"Using the numbers in the list {nums}, "
            f"create an equation that equals {target}."
        )

    prompt = (
        prompt_template.replace("{question}", question)
        if "{question}" in prompt_template
        else f"{prompt_template}\n\n{question}"
    )

    ground_truth = (
        example.get("target")
        or example.get("answer")
        or example.get("gt")
        or 0
    )
    return prompt, ground_truth


# ── vLLM helpers ─────────────────────────────────────────────────────────────

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float):
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_seed

    vllm_set_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
        )


def load_policy_into_vllm(policy: torch.nn.Module, llm) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(
    policy: torch.nn.Module,
    llm,
    prompts: list[str],
    ground_truths: list[Any],
    max_new_tokens: int,
) -> dict[str, float]:
    from vllm import SamplingParams

    load_policy_into_vllm(policy, llm)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
    )

    total = format_total = answer_total = 0.0
    for output, gt in zip(outputs, ground_truths):
        text = output.outputs[0].text
        r = countdown_reward_fn(text, gt)
        total += r["reward"]
        format_total += r["format_reward"]
        answer_total += r["answer_reward"]

    n = max(len(prompts), 1)
    return {
        "reward": total / n,
        "format_reward": format_total / n,
        "answer_reward": answer_total / n,
    }


# ── W&B helpers ───────────────────────────────────────────────────────────────

def init_wandb(args: argparse.Namespace, config: dict) -> None:
    mode = (
        "disabled"
        if args.disable_wandb or not os.environ.get("WANDB_API_KEY")
        else "online"
    )
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=config,
        mode=mode,
    )
    wandb.define_metric("grpo_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="grpo_step")
    wandb.define_metric("eval/*", step_metric="eval_step")


def log(metrics: dict) -> None:
    if wandb.run is not None:
        wandb.log(metrics)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training on Countdown.")
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--train-path", required=True)
    p.add_argument("--val-path", required=True)
    p.add_argument("--test-path", default=None)
    p.add_argument("--prompt-name", default="countdown")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", default=None)

    # GRPO hyperparams
    p.add_argument("--n-grpo-steps", type=int, default=200)
    p.add_argument("--rollout-batch-size", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--sampling-temperature", type=float, default=0.7)
    p.add_argument("--sampling-min-tokens", type=int, default=4)
    p.add_argument("--sampling-max-tokens", type=int, default=1024)
    p.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--advantage-eps", type=float, default=1e-6)
    p.add_argument("--cliprange", type=float, default=0.2)

    # Loss / normalization
    p.add_argument(
        "--loss-type",
        default="reinforce_with_baseline",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    )
    p.add_argument("--use-std-normalization", default="true")
    p.add_argument(
        "--length-normalization",
        default="masked_mean",
        choices=["masked_mean", "masked_normalize"],
    )

    # Optimizer
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)

    # Devices / infra
    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--eval-device", default="cuda:1")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)

    # Eval / logging
    p.add_argument("--eval-every-steps", type=int, default=10)
    p.add_argument("--eval-max-examples", type=int, default=256)
    p.add_argument("--wandb-project", default="grpo-countdown")
    p.add_argument("--wandb-entity", default="saravargasmar-new-york-university")
    p.add_argument("--disable-wandb", action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_device = torch.device(
        args.policy_device if torch.cuda.is_available() else "cpu"
    )

    # ── Batch-size arithmetic ──────────────────────────────────────────────
    rollout_batch_size = args.rollout_batch_size
    group_size = args.group_size
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout = rollout_batch_size // group_size

    # Gracefully handle gradient_accumulation_steps > train_batch_size
    effective_grad_accum = min(args.gradient_accumulation_steps, rollout_batch_size)
    micro_batch_size = max(1, rollout_batch_size // effective_grad_accum)
    n_microbatches = rollout_batch_size // micro_batch_size

    use_std_norm = args.use_std_normalization.lower() == "true"
    normalize_constant = float(args.sampling_max_tokens)  # for masked_normalize

    # ── Model & tokenizer ─────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if policy_device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(policy_device)
    policy.train()

    # ── Datasets ──────────────────────────────────────────────────────────
    prompt_template = load_prompt(args.prompt_name)

    train_ds = load_from_disk(args.train_path)
    val_ds = load_from_disk(args.val_path)
    test_ds = load_from_disk(args.test_path) if args.test_path else None

    train_examples = [extract_countdown_example(ex, prompt_template) for ex in train_ds]
    train_prompts = [e[0] for e in train_examples]
    train_gts = [e[1] for e in train_examples]

    val_examples = [extract_countdown_example(ex, prompt_template) for ex in val_ds]
    val_prompts = [e[0] for e in val_examples[: args.eval_max_examples]]
    val_gts = [e[1] for e in val_examples[: args.eval_max_examples]]

    # ── vLLM ──────────────────────────────────────────────────────────────
    print(f"Initialising vLLM on {args.eval_device}")
    llm = init_vllm(args.model, args.eval_device, args.seed, args.gpu_memory_utilization)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── W&B ───────────────────────────────────────────────────────────────
    config = vars(args).copy()
    config.update(
        {
            "effective_rollout_batch_size": rollout_batch_size,
            "micro_batch_size": micro_batch_size,
            "n_microbatches": n_microbatches,
        }
    )
    init_wandb(args, config)

    # ── Sampling params (rollout) ──────────────────────────────────────────
    from vllm import SamplingParams

    rollout_params = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
    )

    # ── Training loop ─────────────────────────────────────────────────────
    train_history: list[dict] = []
    eval_history: list[dict] = []
    eval_step = 0

    for grpo_step in tqdm(range(1, args.n_grpo_steps + 1), desc="GRPO"):
        policy.train()

        # 1. Sample prompts
        batch_idx = random.sample(range(len(train_prompts)), n_prompts_per_rollout)
        batch_prompts = [train_prompts[i] for i in batch_idx]
        batch_gts = [train_gts[i] for i in batch_idx]

        # 2. Expand by group_size
        repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]
        repeated_gts = [gt for gt in batch_gts for _ in range(group_size)]

        # 3. Generate rollouts
        load_policy_into_vllm(policy, llm)
        vllm_outputs = llm.generate(repeated_prompts, rollout_params)

        rollout_responses: list[str] = []
        for out in vllm_outputs:
            text = out.outputs[0].text
            # Append closing tag if the stop-string was consumed but not included
            if "<answer>" in text and "</answer>" not in text:
                text = text + "</answer>"
            rollout_responses.append(text)

        # 4. Rewards & advantages
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=countdown_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=use_std_norm,
        )
        advantages = advantages.to(policy_device)
        raw_rewards = raw_rewards.to(policy_device)

        # 5. Tokenize full rollout batch once (shared across epochs)
        all_tok = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer
        )
        all_input_ids = all_tok["input_ids"]        # (rollout_batch, L)
        all_labels = all_tok["labels"]
        all_response_masks = all_tok["response_mask"]

        # 6. For grpo_clip: compute old log-probs before any weight update
        all_old_log_probs: torch.Tensor | None = None
        if args.loss_type == "grpo_clip":
            policy.eval()
            chunks = []
            with torch.inference_mode():
                for start in range(0, rollout_batch_size, micro_batch_size):
                    end = min(start + micro_batch_size, rollout_batch_size)
                    out = get_response_log_probs(
                        policy,
                        all_input_ids[start:end].to(policy_device),
                        all_labels[start:end].to(policy_device),
                        return_token_entropy=False,
                    )
                    chunks.append(out["log_probs"].cpu())
            all_old_log_probs = torch.cat(chunks, dim=0)
            policy.train()

        # 7. Inner training loop (epochs × microbatches)
        step_losses: list[float] = []
        step_grad_norms: list[float] = []
        step_entropies: list[float] = []
        step_clip_fracs: list[float] = []

        for epoch in range(args.epochs_per_rollout_batch):
            perm = torch.randperm(rollout_batch_size)
            optimizer.zero_grad(set_to_none=True)

            for mb_i in range(n_microbatches):
                mb_start = mb_i * micro_batch_size
                mb_end = min(mb_start + micro_batch_size, rollout_batch_size)
                idx = perm[mb_start:mb_end]

                mb_input_ids = all_input_ids[idx].to(policy_device)
                mb_labels = all_labels[idx].to(policy_device)
                mb_response_mask = all_response_masks[idx].to(policy_device)
                mb_advantages = advantages[idx].unsqueeze(-1)   # (mb, 1)
                mb_raw_rewards = raw_rewards[idx].unsqueeze(-1)  # (mb, 1)
                mb_old_lp = (
                    all_old_log_probs[idx].to(policy_device)
                    if all_old_log_probs is not None
                    else None
                )

                # Forward pass
                want_entropy = (mb_i == 0 and epoch == 0)
                lp_out = get_response_log_probs(
                    policy, mb_input_ids, mb_labels,
                    return_token_entropy=want_entropy,
                )
                policy_log_probs = lp_out["log_probs"]

                # Per-token loss
                per_token_loss, meta = compute_policy_gradient_loss(
                    policy_log_probs=policy_log_probs,
                    loss_type=args.loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_lp,
                    cliprange=args.cliprange,
                )

                # Aggregate over sequence dimension
                if args.length_normalization == "masked_mean":
                    per_example_loss = masked_mean(
                        per_token_loss, mb_response_mask, dim=-1
                    )
                else:
                    per_example_loss = masked_normalize(
                        per_token_loss,
                        mb_response_mask,
                        normalize_constant=normalize_constant,
                        dim=-1,
                    )

                loss = per_example_loss.mean() / n_microbatches
                loss.backward()
                step_losses.append(loss.item() * n_microbatches)  # un-scaled for logging

                # Collect entropy / clip fraction from first microbatch
                if want_entropy and "token_entropy" in lp_out:
                    ent = masked_mean(
                        lp_out["token_entropy"], mb_response_mask
                    ).item()
                    step_entropies.append(ent)
                if "clip_fraction" in meta:
                    cf = masked_mean(meta["clip_fraction"], mb_response_mask).item()
                    step_clip_fracs.append(cf)

            # Gradient step after all microbatches for this epoch
            grad_norm = clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
            step_grad_norms.append(grad_norm.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 8. Log training metrics
        train_row = {
            "grpo_step": grpo_step,
            "train/loss": sum(step_losses) / max(len(step_losses), 1),
            "train/grad_norm": sum(step_grad_norms) / max(len(step_grad_norms), 1),
            "train/mean_reward": reward_meta["mean_raw_reward"],
            "train/mean_advantage": reward_meta["mean_normalized_reward"],
        }
        if step_entropies:
            train_row["train/token_entropy"] = sum(step_entropies) / len(step_entropies)
        if step_clip_fracs:
            train_row["train/clip_fraction"] = sum(step_clip_fracs) / len(step_clip_fracs)

        # per-rollout reward breakdown
        format_rewards = [
            countdown_reward_fn(r, gt)["format_reward"]
            for r, gt in zip(rollout_responses, repeated_gts)
        ]
        answer_rewards = [
            countdown_reward_fn(r, gt)["answer_reward"]
            for r, gt in zip(rollout_responses, repeated_gts)
        ]
        train_row["train/format_reward"] = sum(format_rewards) / len(format_rewards)
        train_row["train/answer_reward"] = sum(answer_rewards) / len(answer_rewards)

        train_history.append(train_row)
        log(train_row)

        print(
            f"[step {grpo_step:3d}] loss={train_row['train/loss']:.4f}  "
            f"reward={train_row['train/mean_reward']:.3f}  "
            f"ans_acc={train_row['train/answer_reward']:.3f}  "
            f"grad={train_row['train/grad_norm']:.3f}"
        )

        # 9. Periodic evaluation
        if grpo_step % args.eval_every_steps == 0:
            policy.eval()
            eval_step += 1
            val_metrics = evaluate(
                policy, llm, val_prompts, val_gts, args.sampling_max_tokens
            )
            eval_row = {
                "eval_step": eval_step,
                "grpo_step": grpo_step,
                **{f"eval/{k}": v for k, v in val_metrics.items()},
            }
            eval_history.append(eval_row)
            log(eval_row)
            print(
                f"  [eval] reward={val_metrics['reward']:.3f}  "
                f"ans_acc={val_metrics['answer_reward']:.3f}"
            )
            policy.train()

    # ── Final evaluation ──────────────────────────────────────────────────
    policy.eval()
    eval_step += 1
    final_val = evaluate(policy, llm, val_prompts, val_gts, args.sampling_max_tokens)
    results: dict[str, Any] = {
        "grpo_steps": args.n_grpo_steps,
        "val_reward": final_val["reward"],
        "val_answer_reward": final_val["answer_reward"],
        "val_format_reward": final_val["format_reward"],
    }

    if test_ds is not None:
        test_examples = [extract_countdown_example(ex, prompt_template) for ex in test_ds]
        test_prompts = [e[0] for e in test_examples[: args.eval_max_examples]]
        test_gts = [e[1] for e in test_examples[: args.eval_max_examples]]
        test_metrics = evaluate(policy, llm, test_prompts, test_gts, args.sampling_max_tokens)
        results.update(
            {
                "test_reward": test_metrics["reward"],
                "test_answer_reward": test_metrics["answer_reward"],
                "test_format_reward": test_metrics["format_reward"],
            }
        )

    log({"eval_step": eval_step, **{f"eval/{k}": v for k, v in results.items() if isinstance(v, float)}})

    # ── Save ──────────────────────────────────────────────────────────────
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "train_history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_history)
    )
    (output_dir / "eval_history.jsonl").write_text(
        "\n".join(json.dumps(r) for r in eval_history)
    )
    print(json.dumps(results, indent=2))

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
