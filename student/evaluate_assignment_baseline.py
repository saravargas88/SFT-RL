from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from student.drgrpo_grader import question_only_reward_fn


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text().strip()


def build_intellect_examples(dataset) -> tuple[list[str], list[str]]:
    prompts, ground_truths = [], []
    for example in dataset:
        messages = example.get("messages", [])
        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
        prompt = f"{system_msg}\n\n{user_msg}".strip() if system_msg else user_msg
        prompts.append(prompt)
        ground_truths.append(example.get("ground_truth", ""))
    return prompts, ground_truths


def build_math_examples(dataset, prompt_template: str) -> tuple[list[str], list[str]]:
    prompts = [f"{prompt_template}\n\n{example['problem']}" for example in dataset]
    ground_truths = [example["answer"] for example in dataset]
    return prompts, ground_truths


def evaluate(llm: LLM, prompts: list[str], ground_truths: list[str], max_tokens: int) -> tuple[float, list[dict[str, Any]]]:
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)

    correct = 0.0
    results: list[dict[str, Any]] = []
    for output, ground_truth in tqdm(zip(outputs, ground_truths, strict=True), total=len(outputs), desc="Grading"):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truth)
        correct += reward["reward"]
        results.append(
            {
                "output": text,
                "ground_truth": ground_truth,
                "reward": reward["reward"],
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
            }
        )

    accuracy = correct / max(len(outputs), 1)
    return accuracy, results


def summarize_reward_categories(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "correct_format_and_answer": sum(
            1 for result in results if result["format_reward"] == 1 and result["answer_reward"] == 1
        ),
        "correct_format_wrong_answer": sum(
            1 for result in results if result["format_reward"] == 1 and result["answer_reward"] == 0
        ),
        "wrong_format_wrong_answer": sum(
            1 for result in results if result["format_reward"] == 0 and result["answer_reward"] == 0
        ),
        "wrong_format_right_answer": sum(
            1 for result in results if result["format_reward"] == 0 and result["answer_reward"] == 1
        ),
    }


def maybe_limit(dataset, max_examples: int | None):
    if max_examples is None:
        return dataset
    return dataset.select(range(min(max_examples, len(dataset))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot or fine-tuned models on assignment datasets.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", choices=["math", "intellect", "both"], default="math")
    parser.add_argument("--prompt-name", default="intellect")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--math-dataset", default="hiyouga/math12k")
    parser.add_argument("--math-split", default="test")
    parser.add_argument("--intellect-path", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_template = load_prompt(args.prompt_name)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}

    if args.dataset in {"intellect", "both"}:
        if args.intellect_path is None:
            raise ValueError("--intellect-path is required when evaluating the Intellect dataset.")
        dataset = maybe_limit(load_from_disk(args.intellect_path), args.max_examples)
        prompts, ground_truths = build_intellect_examples(dataset)
        accuracy, results = evaluate(llm, prompts, ground_truths, args.max_tokens)
        summaries["intellect"] = {
            "accuracy": accuracy,
            "num_examples": len(results),
            "reward_categories": summarize_reward_categories(results),
        }
        print(json.dumps({"intellect": summaries["intellect"]}, indent=2))
        if output_dir is not None:
            (output_dir / "intellect_results.json").write_text(json.dumps(results, indent=2))

    if args.dataset in {"math", "both"}:
        dataset = maybe_limit(load_dataset(args.math_dataset, split=args.math_split), args.max_examples)
        prompts, ground_truths = build_math_examples(dataset, prompt_template)
        accuracy, results = evaluate(llm, prompts, ground_truths, args.max_tokens)
        summaries["math"] = {
            "accuracy": accuracy,
            "num_examples": len(results),
            "reward_categories": summarize_reward_categories(results),
        }
        print(json.dumps({"math": summaries["math"]}, indent=2))
        if output_dir is not None:
            (output_dir / "math_results.json").write_text(json.dumps(results, indent=2))

    if output_dir is not None:
        (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
