# [REUSED] This file is reused from compute-optimal-tts
# (https://github.com/RyanLiu112/compute-optimal-tts).
# It implements reward model calling functions, PRM special token handling,
# and remote reward inference via FastChat worker API.

import copy
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import requests

from compute_optimal_tts.inference.infer_fns import (
    _step_logit_infer_fn,
    _value_head_infer_fn,
    _chat_logit_infer_fn,
    _chat_logit_v2_infer_fn,
    _step_logit_v2_infer_fn,
    _value_step_infer_fn,
    _special_token_infer_fn,
)


def get_prm_special_tokens(model_type, tokenizer):
    step_tag_id, returned_token_ids = None, None
    if model_type == 'step_logit':
        prm_step_tag = "ки"
        good_tag, bad_tag = "+", "-"
        step_tag_id = tokenizer.encode(prm_step_tag)[-1]
        returned_token_ids = tokenizer.encode(f"{good_tag} {bad_tag}")[1:]
    elif model_type == 'special_token':
        prm_step_tag = "<extra_0>"
        step_tag_id = tokenizer.encode(prm_step_tag)[0]
        returned_token_ids = []
    elif model_type == 'value_head':
        prm_step_tag = "\n"
        step_tag_id = tokenizer.encode(prm_step_tag)[-1]
        returned_token_ids = []
    elif model_type == 'chat_logit':
        good_tag, bad_tag = "+", "-"
        good_tag_id = tokenizer.encode(good_tag)[-1]
        bad_tag_id = tokenizer.encode(bad_tag)[-1]
        returned_token_ids = [good_tag_id, bad_tag_id]
    elif model_type == 'chat_logit_v2':
        good_tag, bad_tag = "+", "-"
        good_tag_id = tokenizer.encode(good_tag)[-1]
        bad_tag_id = tokenizer.encode(bad_tag)[-1]
        returned_token_ids = [good_tag_id, bad_tag_id]
    elif model_type == 'step_logit_v2':
        prm_step_tag = "ки"
        good_tag, bad_tag = "+", "-"
        step_tag_id = tokenizer.encode(f" {prm_step_tag}")[-1]
        good_tag_id = tokenizer.encode(f" {good_tag}")[-1]
        bad_tag_id = tokenizer.encode(f" {bad_tag}")[-1]
        returned_token_ids = [good_tag_id, bad_tag_id]
    elif model_type == 'value_step':
        prm_step_tag = "[PRM]"
        step_tag_id = tokenizer.encode(prm_step_tag, add_special_tokens=False)[-1]
    else:
        raise ValueError("Unknown model_type: {}".format(model_type))
    return step_tag_id, returned_token_ids


INFER_FN_REGISTRY = {
    "step_logit": _step_logit_infer_fn,
    "value_head": _value_head_infer_fn,
    "special_token": _special_token_infer_fn,
    "chat_logit": _chat_logit_infer_fn,
    "chat_logit_v2": _chat_logit_v2_infer_fn,
    "step_logit_v2": _step_logit_v2_infer_fn,
    "value_step": _value_step_infer_fn,
}


def get_infer_fn(model_type, rm_serve_type='fastchat'):
    if model_type in INFER_FN_REGISTRY:
        return INFER_FN_REGISTRY[model_type]
    raise ValueError("Unknown model_type: {}".format(model_type))


@dataclass
class RewardModelBaseConfig:
    prm_step_tag: str
    format_str: str
    rm_serve_type: str
    step_tag_id: int
    returned_token_ids: List[int]


class RewardModelCallingFunction:

    def __init__(self, config: RewardModelBaseConfig):
        self.config = config
        self.prm_step_tag = config.prm_step_tag
        self.format_str = config.format_str

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    def replace_step_tag(self, answer: str):
        if self.prm_step_tag not in answer:
            answer += f" {self.prm_step_tag}"
        splits = answer.split(f" {self.prm_step_tag}")
        splits = [s.strip() for s in splits]
        response = f" {self.prm_step_tag}".join([s for s in splits if s != ""])
        response += f" {self.prm_step_tag}"
        return response


class DummyRewardModelCaller(RewardModelCallingFunction):

    def __init__(self, config: RewardModelBaseConfig):
        super().__init__(config)

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
    ) -> Union[List[int], List[List[int]]]:

        def fn(s):
            steps = s.split(self.prm_step_tag)
            steps = [s for s in steps if s.strip() != ""]
            return list(range(len(steps)))

        if isinstance(question_answer_pairs[0], str):
            return fn(
                self.format_str.format(
                    question=question_answer_pairs[0],
                    answer=self.replace_step_tag(question_answer_pairs[1]),
                )
            )
        else:
            return [
                fn(
                    self.format_str.format(
                        question=s[0],
                        answer=self.replace_step_tag(s[1]),
                    )
                ) for s in question_answer_pairs
            ]


@dataclass
class RemoteRewardModelConfig(RewardModelBaseConfig):
    model_name: str
    controller_addr: str
    multi_gpu: bool


def _reward_inference_fastchat(input_str, model_name, controller_addr="", multi_gpu=False, timeout=0):
    reward = [[0.0]] * (len(input_str) if isinstance(input_str, list) else 1)
    if multi_gpu:
        ret = requests.post(controller_addr + "/get_worker_address", json={"model": model_name})
        worker_addr = ret.json()["address"]
        if not worker_addr:
            raise ValueError("Value Model name {} does not exist.".format(model_name))
    else:
        worker_addr = controller_addr

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {"input_str": input_str}
    response = None
    try:
        if timeout > 0:
            response = requests.post(worker_addr + "/worker_reward_inference", headers=headers, json=gen_params, timeout=timeout)
        else:
            response = requests.post(worker_addr + "/worker_reward_inference", headers=headers, json=gen_params)
        if not response.text:
            raise ValueError("empty response body")
        results = response.json()
        if "reward" in results:
            reward = results["reward"]
        else:
            raise KeyError(f"missing 'reward' in response: keys={list(results.keys())}")
    except Exception as e:
        if response is not None:
            print(f"reward worker response status={getattr(response,'status_code',None)} body_head={response.text[:500]!r}")
        for i in range(len(input_str)):
            print(f'input_str {i}: {input_str[i]}')
        error_info = traceback.format_exc()
        print(f'Error in _reward_inference_fastchat: {error_info}')
        traceback.print_exc()

    return reward


class RMRemoteCaller(RewardModelCallingFunction):

    def __init__(self, config: RemoteRewardModelConfig, tokenizer, model_type: str = ""):
        self.model_name = config.model_name
        self.model_type = model_type
        self.controller_addr = config.controller_addr
        self.tokenizer = tokenizer

        self.prm_step_tag = config.prm_step_tag
        self.step_tag_id = config.step_tag_id
        self.returned_token_ids = config.returned_token_ids

        self.multi_gpu = config.multi_gpu

        super().__init__(config)

    def process_input(self, qa_pairs, model_names, verbose, legal_action=[]):
        if verbose and legal_action:
            print('*' * 8, 'rm_call.py: start legal action', '*' * 8)
            print('*' * 8, legal_action[0]["raw_action"], '*' * 8)
            print('*' * 8, legal_action[0]["action"], '*' * 8)
            print('*' * 8, legal_action[0]["messages"], '*' * 8)
            print('*' * 8, legal_action[0]["stop_str"], '*' * 8)
            print('*' * 8, legal_action[0]["finish_reason"], '*' * 8)
            print('*' * 8, 'rm_call.py: end legal action', '*' * 8)
        if isinstance(qa_pairs[0], str):
            raise ValueError("The input of PRM should be a list of tuples")
        if self.model_type == 'value_head':
            temp_qa_pairs = copy.deepcopy(qa_pairs)
            for i in range(len(temp_qa_pairs)):
                raw_splits = temp_qa_pairs[i][1].split(f" ки\n")
                splits = []
                for s in raw_splits:
                    temp = s.replace("\n", " ").strip()
                    if temp:
                        splits.append(temp)
                if len(splits) == 1:
                    answer = splits[0]
                else:
                    answer = f"\n".join(splits)
                temp_qa_pairs[i] = (temp_qa_pairs[i][0], answer)
            return temp_qa_pairs
        elif self.model_type == 'special_token':
            conversations = []
            temp_qa_pairs = copy.deepcopy(qa_pairs)
            for i in range(len(temp_qa_pairs)):
                conversation = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": temp_qa_pairs[i][0]},
                ]
                assistant_content = ""
                raw_splits = temp_qa_pairs[i][1].split(f" ки\n")
                for j in range(len(raw_splits)):
                    if raw_splits[j].strip() == "":
                        continue
                    text = raw_splits[j].strip()
                    assistant_content += f"{text}<extra_0>"
                conversation.append({"role": "assistant", "content": assistant_content})
                conversations.append(conversation)
            return conversations
        elif self.model_type in ('chat_logit', 'chat_logit_v2'):
            conversations = []
            temp_qa_pairs = copy.deepcopy(qa_pairs)
            for i in range(len(temp_qa_pairs)):
                conversation = []
                raw_splits = temp_qa_pairs[i][1].split(f" ки\n")
                for j in range(len(raw_splits)):
                    if raw_splits[j].strip() == "":
                        continue
                    if j == 0:
                        text = f"{temp_qa_pairs[i][0]} {raw_splits[j].strip()}"
                    else:
                        text = raw_splits[j].strip()
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "<|reserved_special_token_0|>", "role": "assistant"})
                conversations.append(conversation)
            return conversations
        else:
            input_str = []
            for i in range(len(qa_pairs)):
                answer = self.replace_step_tag(qa_pairs[i][1])
                if self.model_type == 'step_logit_v2':
                    answer = answer.replace(" ки\n", " ки")
                elif self.model_type == 'value_step':
                    answer = answer.replace(" ки", " [PRM]")
                format_str = self.format_str.format(question=qa_pairs[i][0], answer=answer)
                input_str.append(format_str)
            return input_str

    def __call__(
        self,
        qa_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
        verbose: Optional[bool] = False,
        local: Optional[bool] = False,
        legal_action: Optional[List[str]] = [],
        process: Optional[bool] = True,
        timeout: Optional[int] = 0,
    ) -> Union[List[int], List[List[int]]]:
        if process:
            input_str = self.process_input(qa_pairs, model_names, verbose=verbose, legal_action=legal_action)
        else:
            input_str = qa_pairs

        if local:
            infer_fn = get_infer_fn(self.model_type, rm_serve_type='fastchat')
            return infer_fn(input_str)

        return _reward_inference_fastchat(
            input_str=input_str, model_name=self.model_name, controller_addr=self.controller_addr, timeout=timeout
        )
