from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import wandb
from datasets import Dataset, load_dataset, load_from_disk
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.drgrpo_grader import question_only_reward_fn
from student.sft import get_response_log_probs, sft_microbatch_train_step, tokenize_prompt_and_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT on math reasoning data.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train-path", required=True, help="Path to Prime Intellect train split on disk.")
    parser.add_argument("--prime-val-path", required=True, help="Path to Prime Intellect validation/test split.")
    parser.add_argument("--prime-test-path", default=None, help="Optional path to Prime Intellect test split.")
    parser.add_argument("--math-val-dataset", default="hiyouga/math12k")
    parser.add_argument("--math-val-split", default="validation")
    parser.add_argument("--math-test-split", default="test")
    parser.add_argument("--prompt-name", default="intellect")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=None, help="Use N unique train examples. Omit for full dataset.")
    parser.add_argument("--eval-max-examples", type=int, default=256)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--wandb-project", default="sft-math")
    parser.add_argument("--wandb-entity", default="saravargasmar-new-york-university")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt(name: str) -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text().strip()


def load_disk_or_hf(path_or_name: str, split: str | None = None) -> Dataset:
    path = Path(path_or_name)
    if path.exists():
        return load_from_disk(str(path))
    if split is None:
        raise ValueError(f"Need a split when loading HF dataset: {path_or_name}")
    return load_dataset(path_or_name, split=split)


def maybe_select(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def extract_prime_example(example: dict[str, Any]) -> tuple[str, str, str | None]:
    messages = example.get("messages", [])
    system_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
    assistant_msg = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
    prompt = f"{system_msg}\n\n{user_msg}".strip() if system_msg else user_msg
    ground_truth = example.get("ground_truth")
    return prompt, assistant_msg, ground_truth


@dataclass
class SFTBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor


def make_sft_collate(tokenizer):
    def collate_fn(examples: list[dict[str, Any]]) -> SFTBatch:
        prompts, outputs = [], []
        for example in examples:
            prompt, response, _ = extract_prime_example(example)
            prompts.append(prompt)
            outputs.append(response)
        tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)
        return SFTBatch(
            input_ids=tokenized["input_ids"],
            labels=tokenized["labels"],
            response_mask=tokenized["response_mask"],
        )

    return collate_fn


def init_wandb(args: argparse.Namespace, model: torch.nn.Module) -> None:
    if args.disable_wandb:
        return
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=vars(args),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")


def log_metrics(metrics: dict[str, Any]) -> None:
    if wandb.run is not None:
        wandb.log(metrics)


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
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
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_with_vllm(policy, llm, prompts: list[str], ground_truths: list[str], max_new_tokens: int) -> float:
    from vllm import SamplingParams

    load_policy_into_vllm_instance(policy, llm)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
    )
    correct = 0.0
    for output, ground_truth in zip(outputs, ground_truths, strict=True):
        text = output.outputs[0].text
        correct += question_only_reward_fn(text, ground_truth)["reward"]
    return correct / max(len(prompts), 1)


@torch.inference_mode()
def evaluate_with_policy(
    policy,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    device: torch.device,
    max_new_tokens: int,
) -> float:
    policy.eval()
    correct = 0.0
    for prompt, ground_truth in tqdm(zip(prompts, ground_truths, strict=True), total=len(prompts), desc="Eval"):
        batch = tokenizer(prompt, return_tensors="pt").to(device)
        generated = policy.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion_ids = generated[0, batch["input_ids"].shape[1]:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        correct += question_only_reward_fn(completion, ground_truth)["reward"]
    policy.train()
    return correct / max(len(prompts), 1)


def build_math_prompts(dataset: Dataset, prompt_template: str) -> tuple[list[str], list[str]]:
    prompts = [f"{prompt_template}\n\n{example['problem']}" for example in dataset]
    ground_truths = [example["answer"] for example in dataset]
    return prompts, ground_truths


def build_prime_prompts(dataset: Dataset) -> tuple[list[str], list[str]]:
    prompts, ground_truths = [], []
    for example in dataset:
        prompt, _, ground_truth = extract_prime_example(example)
        prompts.append(prompt)
        ground_truths.append(ground_truth if ground_truth is not None else "")
    return prompts, ground_truths


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_device = torch.device(args.policy_device if torch.cuda.is_available() else "cpu")
    prompt_template = load_prompt(args.prompt_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if policy_device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(policy_device)
    policy.train()

    train_dataset = maybe_select(load_from_disk(args.train_path), args.train_limit)
    prime_val_dataset = maybe_select(load_from_disk(args.prime_val_path), args.eval_max_examples)
    prime_test_dataset = maybe_select(load_from_disk(args.prime_test_path), args.eval_max_examples) if args.prime_test_path else None
    math_val_dataset = maybe_select(load_disk_or_hf(args.math_val_dataset, args.math_val_split), args.eval_max_examples)
    math_test_dataset = maybe_select(load_disk_or_hf(args.math_val_dataset, args.math_test_split), args.eval_max_examples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=make_sft_collate(tokenizer),
    )

    optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    llm = None
    if args.use_vllm:
        llm = init_vllm(
            model_id=args.model,
            device=args.eval_device,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    init_wandb(args, policy)

    prime_val_prompts, prime_val_gts = build_prime_prompts(prime_val_dataset)
    math_val_prompts, math_val_gts = build_math_prompts(math_val_dataset, prompt_template)

    train_step = 0
    eval_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_train_steps, disable=args.max_train_steps is None, desc="SFT")

    for epoch in range(args.num_epochs):
        for microbatch_idx, batch in enumerate(train_loader):
            input_ids = batch.input_ids.to(policy_device)
            labels = batch.labels.to(policy_device)
            response_mask = batch.response_mask.to(policy_device)

            outputs = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=outputs["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=1.0,
            )

            should_step = (microbatch_idx + 1) % args.gradient_accumulation_steps == 0
            if should_step:
                clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_step += 1
                progress.update(1)

                batch_response_tokens = response_mask.sum().item()
                log_metrics(
                    {
                        "train_step": train_step,
                        "train/loss": loss.item(),
                        "train/per_example_loss": metadata["per_example_loss"].mean().item(),
                        "train/response_tokens": batch_response_tokens,
                        "train/epoch": epoch,
                    }
                )

                if args.eval_every_steps and train_step % args.eval_every_steps == 0:
                    eval_step += 1
                    if llm is not None:
                        prime_val_acc = evaluate_with_vllm(policy, llm, prime_val_prompts, prime_val_gts, args.max_new_tokens)
                        math_val_acc = evaluate_with_vllm(policy, llm, math_val_prompts, math_val_gts, args.max_new_tokens)
                    else:
                        prime_val_acc = evaluate_with_policy(
                            policy, tokenizer, prime_val_prompts, prime_val_gts, policy_device, args.max_new_tokens
                        )
                        math_val_acc = evaluate_with_policy(
                            policy, tokenizer, math_val_prompts, math_val_gts, policy_device, args.max_new_tokens
                        )
                    log_metrics(
                        {
                            "eval_step": eval_step,
                            "eval/prime_val_accuracy": prime_val_acc,
                            "eval/math_val_accuracy": math_val_acc,
                        }
                    )
                    print(
                        f"[eval step {eval_step}] "
                        f"prime_val_acc={prime_val_acc:.4f} math_val_acc={math_val_acc:.4f}"
                    )

                if args.save_every_steps and train_step % args.save_every_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{train_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                if args.max_train_steps is not None and train_step >= args.max_train_steps:
                    break

        if args.max_train_steps is not None and train_step >= args.max_train_steps:
            break

    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    results: dict[str, Any] = {"train_steps": train_step}
    eval_step += 1
    if llm is not None:
        results["prime_val_accuracy"] = evaluate_with_vllm(policy, llm, prime_val_prompts, prime_val_gts, args.max_new_tokens)
        results["math_val_accuracy"] = evaluate_with_vllm(policy, llm, math_val_prompts, math_val_gts, args.max_new_tokens)
    else:
        results["prime_val_accuracy"] = evaluate_with_policy(
            policy, tokenizer, prime_val_prompts, prime_val_gts, policy_device, args.max_new_tokens
        )
        results["math_val_accuracy"] = evaluate_with_policy(
            policy, tokenizer, math_val_prompts, math_val_gts, policy_device, args.max_new_tokens
        )

    if prime_test_dataset is not None:
        prime_test_prompts, prime_test_gts = build_prime_prompts(prime_test_dataset)
        if llm is not None:
            results["prime_test_accuracy"] = evaluate_with_vllm(policy, llm, prime_test_prompts, prime_test_gts, args.max_new_tokens)
        else:
            results["prime_test_accuracy"] = evaluate_with_policy(
                policy, tokenizer, prime_test_prompts, prime_test_gts, policy_device, args.max_new_tokens
            )

    math_test_prompts, math_test_gts = build_math_prompts(math_test_dataset, prompt_template)
    if llm is not None:
        results["math_test_accuracy"] = evaluate_with_vllm(policy, llm, math_test_prompts, math_test_gts, args.max_new_tokens)
    else:
        results["math_test_accuracy"] = evaluate_with_policy(
            policy, tokenizer, math_test_prompts, math_test_gts, policy_device, args.max_new_tokens
        )

    log_metrics({"eval_step": eval_step, **{f"eval/{k}": v for k, v in results.items() if isinstance(v, (float, int))}})
    save_json(output_dir / "results.json", results)
    print(json.dumps(results, indent=2))

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
