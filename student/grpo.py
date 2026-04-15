from __future__ import annotations

from typing import Callable, Literal

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_rewards = torch.tensor(
        [
            reward_fn(response, ground_truth)["reward"]
            for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
        ],
        dtype=torch.float32,
    )
    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=-1, keepdim=True)

    if normalize_by_std:
        group_denoms = grouped_rewards.std(dim=-1, keepdim=True) + advantage_eps
    else:
        group_denoms = torch.ones_like(group_means)

    normalized_rewards = ((grouped_rewards - group_means) / group_denoms).reshape(-1)
    metadata = {
        "mean_raw_reward": raw_rewards.mean().item(),
        "mean_normalized_reward": normalized_rewards.mean().item(),
    }
    return normalized_rewards, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    pg_loss = -(advantages * ratio)
    clipped_pg_loss = -(advantages * clipped_ratio)
    loss = torch.maximum(pg_loss, clipped_pg_loss)
    metadata = {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "clip_fraction": (ratio != clipped_ratio).float(),
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / mask.sum()
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = masked_mean(
        tensor=per_token_loss,
        mask=response_mask,
        dim=-1,
    )
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    metadata["per_example_loss"] = per_example_loss
    return loss, metadata
