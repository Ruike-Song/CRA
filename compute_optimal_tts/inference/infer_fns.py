# [REUSED] This file is reused from compute-optimal-tts
# (https://github.com/RyanLiu112/compute-optimal-tts).
# It implements PRM-specific inference functions for various reward model architectures.

import copy
import torch


@torch.inference_mode()
def _step_logit_infer_fn(input_str: str, model, tokenizer, device, returned_token_ids, step_tag_id):
    returned_token_ids = torch.tensor(returned_token_ids, device=device)
    step_tag_id = torch.tensor(step_tag_id, device=device)

    rewards = []
    if isinstance(input_str, str):
        input_ids = torch.tensor([tokenizer.encode(input_str)], device=device)
        logits = model(input_ids).logits[:, :, returned_token_ids]
        scores = logits.softmax(dim=-1)[:, :, 0]
        step_scores = scores[input_ids == step_tag_id]
        return step_scores
    elif isinstance(input_str, list):
        for prompt in input_str:
            input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            logits = model(input_ids).logits[:, :, returned_token_ids]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores = scores[input_ids == step_tag_id]
            rewards.append(copy.deepcopy(step_scores))

        del input_ids, logits, scores
        torch.cuda.empty_cache()

    return rewards


@torch.inference_mode()
def _value_head_infer_fn(qa_pairs: str, model, tokenizer, device, step_tag_id, step_tag='\n', special_tag_id=None):
    rewards = []
    for qa_pair in qa_pairs:
        question, answer = qa_pair[0], qa_pair[1]
        answer = answer.replace(step_tag, f"<|vision_start|>") + f"<|vision_start|>"

        prompt_ids = tokenizer.encode(tokenizer.bos_token + question + step_tag, return_tensors="pt").squeeze(0).to(device)
        response_ids = tokenizer.encode(answer, return_tensors="pt").squeeze(0).to(device)
        indices = torch.where(response_ids == special_tag_id)
        response_ids[indices] = step_tag_id
        input_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)

        _, _, scores = model(input_ids=input_ids, return_probs=True)
        mask = indices[0] + len(prompt_ids)
        step_scores = scores[0][mask]
        rewards.append(copy.deepcopy(step_scores))

    del input_ids, indices, scores, prompt_ids, response_ids, mask
    torch.cuda.empty_cache()

    return rewards


@torch.inference_mode()
def _chat_logit_infer_fn(conversations: str, model, tokenizer, device, returned_token_ids, special_tag_id=None, verbose=False):
    returned_token_ids = torch.tensor(returned_token_ids, device=device)

    rewards = []
    for conversation in conversations:
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").squeeze(0).to(device)
        indices = torch.where(input_ids == special_tag_id)
        input_ids[indices] = returned_token_ids[0]
        input_ids = input_ids.unsqueeze(0)

        logits = model(input_ids).logits[:, :, returned_token_ids]
        scores = logits.softmax(dim=-1)[0, :, 0]
        mask = indices[0] - 1
        step_scores = scores[mask]
        rewards.append(copy.deepcopy(step_scores))

    del input_ids, logits, scores, mask
    torch.cuda.empty_cache()

    return rewards


@torch.inference_mode()
def _chat_logit_v2_infer_fn(conversations: str, model, tokenizer, device, returned_token_ids, special_tag_id=None, verbose=False):
    returned_token_ids = torch.tensor(returned_token_ids, device=device)

    rewards = []
    for conversation in conversations:
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").squeeze(0).to(device)
        indices = torch.where(input_ids == special_tag_id)
        input_ids[indices] = returned_token_ids[0]
        input_ids = input_ids.unsqueeze(0)

        logits = model(input_ids).logits[:, :, returned_token_ids]
        scores = logits.softmax(dim=-1)[0, :, 0]
        mask = indices[0] - 1
        step_scores = scores[mask]
        rewards.append(copy.deepcopy(step_scores))

    del input_ids, logits, scores, mask
    torch.cuda.empty_cache()

    return rewards


@torch.inference_mode()
def _special_token_infer_fn(conversations: str, model, tokenizer, device, special_tag_id=None, verbose=False):
    rewards = []
    for conversation in conversations:
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
        indices = (input_ids == special_tag_id)

        logits = model(input_ids)[0]
        scores = logits.softmax(dim=-1)[0]
        probabilities = scores * indices[0].unsqueeze(-1)
        step_scores = probabilities[probabilities != 0].view(-1, 2)[:, 1]
        rewards.append(copy.deepcopy(step_scores))

    del input_ids, logits, scores
    torch.cuda.empty_cache()

    return rewards


@torch.inference_mode()
def _step_logit_v2_infer_fn(input_str: str, model, tokenizer, device, returned_token_ids, step_tag_id, verbose=False):
    returned_token_ids = torch.tensor(returned_token_ids, device=device)
    step_tag_id = torch.tensor(step_tag_id, device=device)

    rewards = []
    for prompt in input_str:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        logits = model(input_ids).logits[:, :, returned_token_ids]
        scores = logits.softmax(dim=-1)[:, :, 0]
        step_scores = scores[input_ids == step_tag_id]
        rewards.append(copy.deepcopy(step_scores))

    del input_ids, logits, scores
    torch.cuda.empty_cache()

    return rewards


@torch.inference_mode()
def _value_step_infer_fn(input_str: str, model, tokenizer, device, step_tag_id, verbose=False):
    step_tag_id = torch.tensor(step_tag_id, device=device)

    rewards = []
    for prompt in input_str:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        _, _, scores = model(input_ids=input_ids)
        step_scores = scores[input_ids == step_tag_id]
        rewards.append(copy.deepcopy(step_scores))

    del input_ids, scores
    torch.cuda.empty_cache()

    return rewards
