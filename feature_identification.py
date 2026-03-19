#!/usr/bin/env python3
"""
Identifying Reward Hacking Semantics
=====================================

Use two-sample t-tests to find SAE latent features that discriminate between
reward-hacking steps and normal reasoning steps.

For each feature j we compute:
    t_j = (mu_1_j - mu_0_j) / sqrt(sigma2_1_j / n_1 + sigma2_0_j / n_0)

Selected features satisfy:
    |t_j| > tau_t   AND   max(mu_1_j, mu_0_j) > tau_a
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from tqdm import tqdm
import warnings
import glob

from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAE model (lightweight, for loading a trained checkpoint)
# ---------------------------------------------------------------------------

class SimpleSAE(nn.Module):
    """Minimal SAE used to load a pre-trained checkpoint."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder_weight = nn.Parameter(torch.randn(d_hidden, d_model))
        self.encoder_bias = nn.Parameter(torch.zeros(d_hidden))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.encoder_weight.T + self.encoder_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.encoder_weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ---------------------------------------------------------------------------
# Statistical Feature Identifier
# ---------------------------------------------------------------------------

class StatisticalFeatureIdentifier:
    """
    Compute two-sample t-statistics for every SAE feature dimension and
    rank them to identify reward-hacking confounders.
    """

    def identify_confounding_features(
        self,
        hack_features: np.ndarray,
        normal_features: np.ndarray,
    ) -> Dict:
        """
        Parameters
        ----------
        hack_features : ndarray of shape [n_1, m]
            SAE features for reward-hacking steps.
        normal_features : ndarray of shape [n_0, m]
            SAE features for normal steps.

        Returns
        -------
        dict with sorted feature statistics.
        """
        logger.info("Computing two-sample t-statistics for all features ...")

        hack_clean, normal_clean = self._preprocess(hack_features, normal_features)
        if hack_clean is None:
            return {"status": "failed", "reason": "insufficient data quality"}

        n_1, m = hack_clean.shape
        n_0 = normal_clean.shape[0]

        all_features = []
        for j in tqdm(range(m), desc="Feature statistics"):
            hv = hack_clean[:, j]
            nv = normal_clean[:, j]

            mu_1 = np.mean(hv)
            mu_0 = np.mean(nv)
            s2_1 = np.var(hv, ddof=1)
            s2_0 = np.var(nv, ddof=1)

            denom = np.sqrt(s2_1 / n_1 + s2_0 / n_0)
            t_j = (mu_1 - mu_0) / denom if denom > 0 else 0.0

            all_features.append(
                {
                    "feature_idx": j,
                    "t_statistic": float(t_j),
                    "abs_t_statistic": float(abs(t_j)),
                    "mu_1_j": float(mu_1),
                    "mu_0_j": float(mu_0),
                    "sigma2_1_j": float(s2_1),
                    "sigma2_0_j": float(s2_0),
                    "mean_difference": float(mu_1 - mu_0),
                    "hack_activation_rate": float((hv > 0).mean()),
                    "normal_activation_rate": float((nv > 0).mean()),
                }
            )

        all_features.sort(key=lambda x: x["t_statistic"], reverse=True)

        results = {
            "status": "success",
            "method": "two_sample_t_test",
            "data_summary": {
                "cleaned_hack_samples": n_1,
                "cleaned_normal_samples": n_0,
                "feature_dimensions": m,
            },
            "statistics_summary": {
                "total_features": len(all_features),
                "mean_t": float(np.mean([f["t_statistic"] for f in all_features])),
                "max_t": float(max(f["t_statistic"] for f in all_features)),
                "min_t": float(min(f["t_statistic"] for f in all_features)),
            },
            "all_features": all_features,
        }
        self._print_summary(results)
        return results

    @staticmethod
    def _preprocess(
        hack: np.ndarray, normal: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if hack.shape[1] != normal.shape[1] or len(hack) < 2 or len(normal) < 2:
            return None, None
        hack = hack[~np.isnan(hack).any(axis=1)]
        normal = normal[~np.isnan(normal).any(axis=1)]
        hack = hack[~np.isinf(hack).any(axis=1)]
        normal = normal[~np.isinf(normal).any(axis=1)]
        if len(hack) < 2 or len(normal) < 2:
            return None, None
        return hack, normal

    @staticmethod
    def _print_summary(results: Dict):
        print("\n" + "=" * 70)
        print("Two-sample t-statistic feature ranking")
        print("=" * 70)
        ds = results["data_summary"]
        print(
            f"  Hack samples: {ds['cleaned_hack_samples']}, "
            f"Normal samples: {ds['cleaned_normal_samples']}, "
            f"Features: {ds['feature_dimensions']}"
        )
        ss = results["statistics_summary"]
        print(
            f"  Total features: {ss['total_features']}, "
            f"max t={ss['max_t']:.4f}, min t={ss['min_t']:.4f}"
        )
        print("\nTop-10 features (by t-statistic, descending):")
        for i, f in enumerate(results["all_features"][:10]):
            print(
                f"  {i + 1:2d}. feature {f['feature_idx']:5d}: "
                f"t={f['t_statistic']:7.3f}, "
                f"mean_diff={f['mean_difference']:+.4f}, "
                f"hack_rate={f['hack_activation_rate']:.3f}, "
                f"normal_rate={f['normal_activation_rate']:.3f}"
            )


# ---------------------------------------------------------------------------
# Feature Identification Experiment
# ---------------------------------------------------------------------------

class FeatureIdentificationExperiment:
    """
    End-to-end pipeline:
      1. Load labelled reasoning steps (reward-hacking vs normal).
      2. Extract PRM activations and encode through a trained SAE.
      3. Run t-test feature ranking.
    """

    def __init__(self, args):
        self.args = args
        self.identifier = StatisticalFeatureIdentifier()

    def run(self) -> Optional[Dict]:
        feature_data = self._extract_feature_data()
        if feature_data is None:
            return None

        torch.cuda.empty_cache()

        results = self.identifier.identify_confounding_features(
            feature_data["hack_features"], feature_data["normal_features"]
        )

        torch.cuda.empty_cache()

        self._save_results(results)
        return results

    # ------------------------------------------------------------------

    def _extract_feature_data(self) -> Optional[Dict]:
        analysis_dir = self.args.analysis_dir
        if not os.path.exists(analysis_dir):
            logger.error(f"Analysis directory not found: {analysis_dir}")
            return None

        hack_steps, normal_steps = [], []

        for fp in tqdm(
            list(Path(analysis_dir).glob("*.json")), desc="Loading analysis files"
        ):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not data.get("has_reward_hacking", False):
                    continue

                problem = data.get("problem", "")
                analyses = data.get("step_by_step_analyses", [])
                for sa in analyses:
                    step_num = sa.get("step_num", 0)
                    cumulative = []
                    for prev in analyses:
                        if prev.get("step_num", 0) <= step_num:
                            c = prev.get("content", "")
                            if c and not c.strip().endswith("\u043a\u0438"):
                                c += " \u043a\u0438"
                            cumulative.append(c)
                    if len(cumulative) > 1:
                        body = "\n".join(cumulative[:-1]) + "\n" + cumulative[-1]
                    elif cumulative:
                        body = cumulative[0]
                    else:
                        body = ""

                    step_info = {
                        "text": f"{problem} {body}",
                        "is_reward_hacking": sa.get("is_reward_hacking", False),
                    }
                    (hack_steps if step_info["is_reward_hacking"] else normal_steps).append(
                        step_info
                    )
            except Exception as e:
                logger.warning(f"Skipping {fp}: {e}")

        if not hack_steps or not normal_steps:
            logger.error("Insufficient labelled steps.")
            return None

        logger.info(
            f"Loaded {len(hack_steps)} hack steps and {len(normal_steps)} normal steps."
        )

        sae_model, _ = self._load_sae()
        if sae_model is None:
            return None

        hack_feats = self._extract_sae_features(hack_steps, sae_model)
        torch.cuda.empty_cache()
        normal_feats = self._extract_sae_features(normal_steps, sae_model)
        torch.cuda.empty_cache()

        if hack_feats is None or normal_feats is None:
            return None

        return {"hack_features": hack_feats, "normal_features": normal_feats}

    def _load_sae(self) -> Tuple[Optional[SimpleSAE], Optional[Dict]]:
        path = self.args.sae_checkpoint
        if not os.path.exists(path):
            logger.error(f"SAE checkpoint not found: {path}")
            return None, None
        ckpt = torch.load(path, map_location="cuda", weights_only=False)
        if not isinstance(ckpt, dict) or "d_model" not in ckpt:
            logger.error("Unrecognized checkpoint format.")
            return None, None
        cfg = {
            "d_model": ckpt["d_model"],
            "d_hidden": ckpt["d_sae"],
        }
        sae = SimpleSAE(cfg["d_model"], cfg["d_hidden"])
        sae.load_state_dict(ckpt["model_state_dict"])
        sae.to("cuda").eval()
        return sae, cfg

    def _extract_sae_features(
        self, steps: List[Dict], sae_model: SimpleSAE
    ) -> Optional[np.ndarray]:
        prm_path = self.args.prm_model_path
        tokenizer = AutoTokenizer.from_pretrained(prm_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prm = AutoModelForCausalLM.from_pretrained(
            prm_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to("cuda")
        prm.eval()

        target_layer_idx = self.args.target_layer
        cache = []

        def hook_fn(module, inp, out):
            cache.append((out[0] if isinstance(out, tuple) else out).detach())

        handle = prm.model.layers[target_layer_idx].register_forward_hook(hook_fn)
        all_act = []

        try:
            for step in tqdm(steps, desc="Extracting activations"):
                cache.clear()
                ids = tokenizer(
                    step["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.args.max_length,
                ).to("cuda")
                with torch.no_grad():
                    _ = prm(**ids)
                if cache:
                    seq_len = ids["input_ids"].shape[1]
                    all_act.append(cache[0][0, seq_len - 1, :].unsqueeze(0).cpu())
                del ids
                torch.cuda.empty_cache()
        finally:
            handle.remove()

        if not all_act:
            return None

        act_tensor = torch.cat(all_act, dim=0).to("cuda")
        with torch.no_grad():
            feats = sae_model.encode(act_tensor)
        result = feats.cpu().numpy()
        del act_tensor, feats
        torch.cuda.empty_cache()
        return result

    def _save_results(self, results: Dict):
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        out = {
            "experiment_info": {
                "method": "statistical_confounding_feature_identification",
                "timestamp": datetime.now().isoformat(),
                "formula": "t_j = (mu_1_j - mu_0_j) / sqrt(sigma2_1_j/n_1 + sigma2_0_j/n_0)",
                "selection": "|t_j| > tau_t AND max(mu_1_j, mu_0_j) > tau_a",
            },
            "identification_results": _convert(results),
        }
        out_path = self.args.output_path
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify reward-hacking features via t-test"
    )
    parser.add_argument(
        "--analysis-dir", type=str, required=True,
        help="Directory containing per-problem JSON analysis files"
    )
    parser.add_argument(
        "--sae-checkpoint", type=str, required=True,
        help="Path to trained SAE checkpoint (.pt)"
    )
    parser.add_argument(
        "--prm-model-path", type=str, required=True,
        help="Path to PRM (reward model) for activation extraction"
    )
    parser.add_argument(
        "--target-layer", type=int, required=True,
        help="Transformer block index to extract activations from"
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Tokenizer max sequence length"
    )
    parser.add_argument(
        "--output-path", type=str, default="statistical_confounding_features.json",
        help="Output JSON path"
    )
    args = parser.parse_args()

    torch.cuda.empty_cache()
    exp = FeatureIdentificationExperiment(args)
    results = exp.run()

    if results and results["status"] == "success":
        n = results["statistics_summary"]["total_features"]
        print(f"\nDone. Computed t-statistics for {n} features.")
    else:
        print("\nExperiment failed.")


if __name__ == "__main__":
    main()
