# [REUSED] This file is reused from compute-optimal-tts
# (https://github.com/RyanLiu112/compute-optimal-tts).
# It implements PreTrainedModelWrapper for wrapping pretrained models with value heads.

# Copyright 2022 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import json
import logging
import os
from copy import deepcopy
from typing import Optional
import sys

import torch
import torch.nn as nn
from accelerate import PartialState
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from safetensors.torch import load_file as safe_load_file
from transformers import PreTrainedModel

if sys.version_info < (3, 8):
    _is_python_greater_3_8 = False
else:
    _is_python_greater_3_8 = True


def is_transformers_greater_than(current_version: str) -> bool:
    if _is_python_greater_3_8:
        from importlib.metadata import version
        _transformers_version = version("transformers")
    else:
        import pkg_resources
        _transformers_version = pkg_resources.get_distribution("transformers").version
    return _transformers_version > current_version


if is_transformers_greater_than("4.33.0"):
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
else:
    from transformers.deepspeed import is_deepspeed_zero3_enabled

LAYER_PATTERNS = [
    "transformer.h.{layer}",
    "model.decoder.layers.{layer}",
    "gpt_neox.layers.{layer}",
    "model.layers.{layer}",
]


class PreTrainedModelWrapper(nn.Module):
    transformers_parent_class = None
    supported_args = None
    supported_modules = ("v_head",)
    supported_rm_modules = ("score",)
    supported_pretrained_model_architectures = ((PreTrainedModel))

    def __init__(
        self, pretrained_model=None, score_module=None, supports_rm_adapter=False, rm_adapter_name=None, **kwargs
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        self.prepare_inputs_for_generation = pretrained_model.prepare_inputs_for_generation
        self.is_loaded_in_8bit = getattr(pretrained_model, "is_loaded_in_8bit", False)
        self.is_loaded_in_4bit = getattr(pretrained_model, "is_loaded_in_4bit", False)
        self.is_sequential_parallel = False

        if hasattr(pretrained_model, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable = pretrained_model.gradient_checkpointing_disable
        if hasattr(pretrained_model, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable = pretrained_model.gradient_checkpointing_enable
        if hasattr(pretrained_model, "enable_input_require_grads"):
            self.enable_input_require_grads = pretrained_model.enable_input_require_grads

        self.supports_rm_adapter = supports_rm_adapter
        self.rm_adapter_name = rm_adapter_name
        self.policy_adapter_name = "default"
        if score_module is not None:
            self.score = score_module

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if kwargs is not None:
            peft_config = kwargs.pop("peft_config", None)
            reward_adapter = kwargs.pop("reward_adapter", None)
            reward_adapter_name = kwargs.pop("reward_adapter_name", "reward_adapter")
            is_trainable = kwargs.pop("is_trainable", False)
            trl_model_args, pretrained_kwargs, peft_quantization_kwargs = cls._split_kwargs(kwargs)
            token = pretrained_kwargs.get("token", None)
        else:
            peft_config = None
            is_trainable = False
            trl_model_args = {}
            pretrained_kwargs = {}
            peft_quantization_kwargs = {}
            token = None

        if reward_adapter is not None and not isinstance(reward_adapter, str):
            raise ValueError(
                "The `reward_adapter` argument should be a string."
            )

        is_peft_model = False
        current_device = cls._get_current_device()

        if isinstance(pretrained_model_name_or_path, str):
            is_loaded_in_8bit = pretrained_kwargs.get("load_in_8bit", False)
            is_loaded_in_4bit = pretrained_kwargs.get("load_in_4bit", False)
        else:
            is_loaded_in_8bit = getattr(pretrained_model_name_or_path, "is_loaded_in_8bit", False)
            is_loaded_in_4bit = getattr(pretrained_model_name_or_path, "is_loaded_in_4bit", False)

        if (is_loaded_in_8bit or is_loaded_in_4bit) and "device_map" not in pretrained_kwargs:
            logging.warning("The `device_map` argument is not provided. Overriding to current device.")
            pretrained_kwargs["device_map"] = {"": current_device}

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, **pretrained_kwargs
            )
        elif isinstance(pretrained_model_name_or_path, cls.supported_pretrained_model_architectures):
            pretrained_model = pretrained_model_name_or_path
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string or a PreTrainedModel"
            )

        if not is_peft_model and reward_adapter is not None:
            raise ValueError("reward_adapter can only be used with a PeftModel.")

        multi_adapter_args = {"supports_rm_adapter": False}
        model = cls(pretrained_model, **multi_adapter_args, **trl_model_args)

        is_resuming_training = True
        if isinstance(pretrained_model_name_or_path, str):
            safe_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            sharded_index_filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin.index.json")
            safe_sharded_index_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
            is_sharded = False
            use_safe = os.path.exists(safe_filename)

            if not (os.path.exists(filename) or os.path.exists(safe_filename)):
                filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                    pretrained_model, pretrained_model_name_or_path, sharded_index_filename, token=token,
                )
                if filename is None and files_to_download is None:
                    safe_filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                        pretrained_model, pretrained_model_name_or_path, safe_sharded_index_filename, token=token,
                        model_name="model.safetensors", model_index_name="model.safetensors.index.json",
                    )
                    use_safe = True
                else:
                    use_safe = False

            loading_func = safe_load_file if use_safe else torch.load
            load_kwargs = {} if use_safe else {"map_location": "cpu"}

            if is_resuming_training:
                if is_sharded:
                    state_dict = {}
                    for shard_file in files_to_download:
                        filename = hf_hub_download(pretrained_model_name_or_path, shard_file, token=token)
                        state_dict.update(loading_func(filename, **load_kwargs))
                else:
                    state_dict = loading_func(filename if not use_safe else safe_filename, **load_kwargs)
        else:
            state_dict = pretrained_model_name_or_path.state_dict()

        model.is_peft_model = is_peft_model
        model.current_device = current_device

        if is_resuming_training:
            model.post_init(state_dict=state_dict)

        return model

    @classmethod
    def _get_checkpoint_from_hub(
        cls, pretrained_model, pretrained_model_name_or_path, index_filename, token=None,
        model_name="pytorch_model.bin", model_index_name="pytorch_model.bin.index.json",
    ):
        files_to_download = None
        filename = None
        is_resuming_training = True
        is_sharded = False

        try:
            filename = hf_hub_download(pretrained_model_name_or_path, model_name, token=token)
        except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
            if os.path.exists(index_filename):
                index_file_name = index_filename
            else:
                try:
                    index_file_name = hf_hub_download(pretrained_model_name_or_path, model_index_name, token=token)
                except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
                    is_resuming_training = False
                    logging.warning(
                        f"A {type(pretrained_model)} model is loaded from '{pretrained_model_name_or_path}', "
                        f"and no v_head weight is found."
                    )
            if is_resuming_training:
                with open(index_file_name) as f:
                    index = json.load(f)
                files_to_download = set()
                for k, v in index["weight_map"].items():
                    if any(module in k for module in cls.supported_modules):
                        files_to_download.add(v)
                is_sharded = True

        return filename, files_to_download, is_sharded, is_resuming_training

    @classmethod
    def _get_current_device(cls):
        state = PartialState()
        return state.local_process_index if torch.cuda.is_available() else "cpu"

    @classmethod
    def _split_kwargs(cls, kwargs):
        supported_kwargs = {}
        unsupported_kwargs = {}
        peft_kwargs = {}

        for key, value in kwargs.items():
            if key in cls.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value

        return supported_kwargs, unsupported_kwargs, peft_kwargs

    def push_to_hub(self, *args, **kwargs):
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        state_dict = kwargs.get("state_dict")
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict

        if self.is_peft_model:
            save_path = args[0]
            save_path = os.path.join(save_path, "pytorch_model.bin")
            torch.save(state_dict, save_path)
            _ = kwargs.pop("state_dict", None)

        return self.pretrained_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        raise NotImplementedError

    def post_init(self, *args, **kwargs):
        raise NotImplementedError
