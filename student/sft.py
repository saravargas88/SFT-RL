from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


'''
SFT : we are initializing a model and then fine tuning it on a dataset of 

(prompt, response) pairs

We are then going to take n steps of sampling a batch of pairs 
compute the crossentropy loss between model output and response output 
update the model parameteres taking the gradient step 


'''
#1: TOKENIZE AND CONSTRUCT RESPONSE MASK : 
# we have to tokenize the prompt and response separately because we need to know 
# where the prompt ends and response starts in order to construct the response mask 

# The response mask is a boolean tensor that tells us which tokens in the input 
# are part of the response and which are part of the prompt


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    
    '''
    First tokenize the prompt and output the strings separately
    Then join them together and construct a response_mask
    Returns: disctionary of keys: input_ids, labels, response_mask
    '''
    
    #1. tokenize the prompt and output strings separately
    prompt_tokens = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_tokens = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    

    #2. response mask 
    prompt_lens = torch.tensor([len(ids) for ids in prompt_tokens], dtype=torch.long)
    
    concatenated_ids = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_tokens, output_tokens, strict=True)
    ] #join the prompt and output tokens together for each pair of strings
    
    total_lens = [len(ids) for ids in concatenated_ids]
    max_total_len = max(total_lens)

    #padded tensor for input id attention mask is response mask
    pad_token_id = tokenizer.pad_token_id
    
    full_input_ids = torch.full(
        (len(concatenated_ids), max_total_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    for i, ids in enumerate(concatenated_ids):
        full_input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    input_ids = full_input_ids[:, :-1]
    labels    = full_input_ids[:, 1:]

    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, (plen, seq) in enumerate(zip(prompt_lens, concatenated_ids)):
        response_mask[i, plen - 1 : len(seq) - 1] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy over the vocabulary dimension for each token position."""
    
    
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Score labels under a causal LM and optionally return per-token entropy."""
    logits = model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    output = {"log_probs": token_log_probs}
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)
    return output


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum masked elements and divide by a provided normalization constant."""
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / normalize_constant
    return masked_tensor.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute and backprop the masked SFT loss for one microbatch."""
    per_example_loss = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1,
    )
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {"per_example_loss": per_example_loss}
