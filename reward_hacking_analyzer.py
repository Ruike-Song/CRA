#!/usr/bin/env python3
"""
Reward Hacking Analyzer
=======================

Utility module providing:
- SimpleSAE: lightweight SAE class matching saved checkpoint format
- MathShepherdActivationExtractor: batch activation extraction from PRM
- RewardHackingAnalyzer: load labelled data and extract SAE features

Used by feature identification and backdoor adjustment pipelines.
"""

import os
import json
import glob
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSAE(nn.Module):
    """SAE model matching the saved checkpoint layout (tied weights)."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.encoder_weight = nn.Parameter(torch.randn(d_hidden, d_model))
        self.encoder_bias = nn.Parameter(torch.zeros(d_hidden))
        self.relu = nn.ReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x @ self.encoder_weight.T + self.encoder_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.encoder_weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class MathShepherdActivationExtractor:
    """Extract token-level activations from a PRM."""

    def __init__(self, model_path: str, target_layer: int = None, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.target_layer = target_layer
        self.model = None
        self.tokenizer = None

    def load_model(self) -> bool:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False

    def extract_activations(
        self,
        texts: List[str],
        target_layer: int = None,
        batch_size: int = 4,
        max_length: int = 2048,
    ) -> torch.Tensor:
        all_activations: List[torch.Tensor] = []
        cache: List[torch.Tensor] = []

        def hook_fn(module, inp, out):
            cache.append((out[0] if isinstance(out, tuple) else out).detach().cpu())

        layer_module = self.model.model.layers[target_layer]
        handle = layer_module.register_forward_hook(hook_fn)

        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
                batch = texts[i : i + batch_size]
                cache.clear()
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)
                with torch.no_grad():
                    _ = self.model(**inputs)
                if cache:
                    act = cache[0]
                    for j in range(len(batch)):
                        seq_len = (
                            inputs["input_ids"][j] != self.tokenizer.pad_token_id
                        ).sum().item()
                        all_activations.append(act[j, seq_len - 1, :].unsqueeze(0))
        finally:
            handle.remove()

        if all_activations:
            return torch.cat(all_activations, dim=0)
        d = self.model.config.hidden_size
        return torch.empty(0, d)


class RewardHackingAnalyzer:
    """Load labelled reward-hacking data and extract SAE features."""

    def __init__(self, analysis_dir: str, sae_model_path: str, target_layer: int = None, device: str = "cpu"):
        self.analysis_dir = analysis_dir
        self.sae_model_path = sae_model_path
        self.device = device
        self.target_layer = target_layer

    def load_sae_model(self) -> Tuple[Optional[SimpleSAE], Optional[Dict]]:
        if not os.path.exists(self.sae_model_path):
            return None, None
        try:
            ckpt = torch.load(self.sae_model_path, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict):
                return None, None
            if "d_model" in ckpt and "d_sae" in ckpt:
                cfg = {"d_model": ckpt["d_model"], "d_hidden": ckpt["d_sae"]}
                state = ckpt["model_state_dict"]
            elif "config" in ckpt:
                cfg = ckpt["config"]
                state = ckpt["model_state_dict"]
            else:
                keys = list(ckpt.keys())
                if "encoder_weight" in keys:
                    d_hidden, d_model = ckpt["encoder_weight"].shape
                    cfg = {"d_model": d_model, "d_hidden": d_hidden}
                else:
                    raise ValueError("Cannot infer SAE dimensions from checkpoint.")
                state = ckpt
            sae = SimpleSAE(cfg["d_model"], cfg["d_hidden"])
            sae.load_state_dict(state)
            sae.to(self.device).eval()
            return sae, cfg
        except Exception as e:
            logger.error(f"SAE loading failed: {e}")
            return None, None

    def load_reward_hacking_data(self) -> Dict[str, List[Dict]]:
        if not os.path.exists(self.analysis_dir):
            return {"reward_hack_steps": [], "normal_steps": [], "all_steps": []}

        hack, normal, all_steps = [], [], []
        for fp in tqdm(
            glob.glob(os.path.join(self.analysis_dir, "*.json")),
            desc="Loading data",
        ):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not data.get("has_reward_hacking", False):
                    continue
                problem = data.get("problem", "")
                for sa in data.get("step_by_step_analyses", []):
                    cur_num = sa["step_num"]
                    prev = [
                        s for s in data["step_by_step_analyses"]
                        if s.get("step_num", 0) <= cur_num
                    ]
                    body = ""
                    for s in prev:
                        if body:
                            body += " \u043a\u0438 "
                        body += s["content"]
                    if not body.endswith(" \u043a\u0438"):
                        body += " \u043a\u0438"
                    info = {
                        "case_id": os.path.basename(fp),
                        "step_num": sa["step_num"],
                        "text": f"{problem} {body}",
                        "step_content": sa["content"],
                        "reward": sa["reward"],
                        "is_reward_hacking": sa.get("is_reward_hacking", False),
                        "problem": problem,
                    }
                    (hack if info["is_reward_hacking"] else normal).append(info)
                    all_steps.append(info)
            except Exception:
                continue

        return {"reward_hack_steps": hack, "normal_steps": normal, "all_steps": all_steps}

    def analyze_feature_differences(
        self, enhanced_data: List[Dict], p_threshold: float = 0.05, diff_threshold: float = 0.0
    ) -> Optional[Dict]:
        from scipy import stats as sp_stats

        hack_f, normal_f = [], []
        for s in enhanced_data:
            (hack_f if s["is_reward_hacking"] else normal_f).append(s["sae_features"])

        if not hack_f or not normal_f:
            return None

        hack_arr = np.array(hack_f)
        normal_arr = np.array(normal_f)
        mean_diff = hack_arr.mean(0) - normal_arr.mean(0)

        significant = []
        for i in range(len(mean_diff)):
            t_stat, p_val = sp_stats.ttest_ind(hack_arr[:, i], normal_arr[:, i])
            if p_val < p_threshold and abs(mean_diff[i]) > diff_threshold:
                significant.append(
                    {
                        "feature_idx": i,
                        "mean_diff": mean_diff[i],
                        "t_stat": t_stat,
                        "p_val": p_val,
                        "hack_mean": hack_arr[:, i].mean(),
                        "normal_mean": normal_arr[:, i].mean(),
                    }
                )

        significant.sort(key=lambda x: abs(x["mean_diff"]), reverse=True)
        return {
            "significant_features": significant,
            "hack_mean": hack_arr.mean(0),
            "normal_mean": normal_arr.mean(0),
            "mean_diff": mean_diff,
        }
