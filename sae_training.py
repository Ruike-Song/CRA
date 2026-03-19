#!/usr/bin/env python3
"""
Training Sparse Autoencoders on Reward Models
===============================================

Train layer-wise SAEs on PRM internal activations to decompose them into
interpretable sparse features.

Loss: L = (1/d)||h - h_hat||^2 + alpha * ||z||_1
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class StandardSAE(nn.Module):
    """
    Standard Sparse Autoencoder with tied weights.

    Encoder: z = ReLU(W_e * h + b_e)
    Decoder: h_hat = W_d * z,  where W_d = W_e^T
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        self.encoder_weight = nn.Parameter(
            torch.randn(d_sae, d_model) / np.sqrt(d_model)
        )
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae))

        self._normalize_decoder()

    def _normalize_decoder(self):
        """Normalize decoder column vectors (transpose of encoder weights)."""
        with torch.no_grad():
            decoder_weight = self.encoder_weight.T
            decoder_weight = decoder_weight / (
                decoder_weight.norm(dim=0, keepdim=True) + 1e-8
            )
            self.encoder_weight.data = decoder_weight.T

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.encoder_weight.T + self.encoder_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.encoder_weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class ActivationDataset:
    """
    Extract token-level activations from a reward model's transformer blocks.
    """

    def __init__(
        self,
        model_path: str,
        target_layer: int,
        use_ddp: bool = True,
        local_rank: int = 0,
        gpu_id: int = None,
    ):
        self.model_path = model_path
        self.target_layer = target_layer
        self.use_ddp = use_ddp
        self.local_rank = local_rank

        if torch.cuda.is_available():
            if use_ddp:
                self.device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(local_rank)
            else:
                self.device = (
                    torch.device(f"cuda:{gpu_id}")
                    if gpu_id is not None
                    else torch.device("cuda")
                )
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(self.device)

        if self.use_ddp and dist.is_initialized():
            self.model = DDP(
                self.model, device_ids=[local_rank], output_device=local_rank
            )

        self.model.eval()

    def extract_activations_from_texts(
        self, texts: List[str], batch_size: int = 4, max_length: int = 512
    ) -> torch.Tensor:
        """Extract activations from a list of reasoning-path texts."""
        all_activations = []
        is_main = (
            not self.use_ddp or not dist.is_initialized() or dist.get_rank() == 0
        )

        activations_cache = []

        def activation_hook(module, input, output):
            if isinstance(output, tuple):
                activations_cache.append(output[0].detach().cpu())
            else:
                activations_cache.append(output.detach().cpu())

        if hasattr(self.model, "module"):
            target_layer = self.model.module.model.layers[self.target_layer]
        else:
            target_layer = self.model.model.layers[self.target_layer]

        hook_handle = target_layer.register_forward_hook(activation_hook)

        if self.use_ddp and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            texts_per_process = len(texts) // world_size
            start_idx = rank * texts_per_process
            end_idx = len(texts) if rank == world_size - 1 else start_idx + texts_per_process
            local_texts = texts[start_idx:end_idx]
        else:
            local_texts = texts

        try:
            for i in tqdm(
                range(0, len(local_texts), batch_size),
                desc="Extracting activations",
                disable=not is_main,
            ):
                batch_texts = local_texts[i : i + batch_size]
                activations_cache.clear()

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)

                with torch.no_grad():
                    _ = self.model(**inputs)

                if activations_cache:
                    batch_act = activations_cache[0]
                    flat = batch_act.view(-1, batch_act.shape[-1])
                    mask = inputs["attention_mask"].cpu().view(-1)
                    all_activations.append(flat[mask.bool()])
        finally:
            hook_handle.remove()

        if all_activations:
            local_result = torch.cat(all_activations, dim=0)
        else:
            d = (
                self.model.config.hidden_size
                if not hasattr(self.model, "module")
                else self.model.module.config.hidden_size
            )
            local_result = torch.empty(0, d)

        if self.use_ddp and dist.is_initialized():
            all_results = [
                torch.empty_like(local_result) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(all_results, local_result)
            if is_main:
                return torch.cat(all_results, dim=0)
            return local_result
        return local_result

    def save_activations(self, activations: torch.Tensor, save_path: str):
        torch.save(activations, save_path)

    def load_activations(self, load_path: str) -> Optional[torch.Tensor]:
        if not os.path.exists(load_path):
            return None
        return torch.load(load_path, map_location="cpu")


class SAETrainer:
    """Train a layer-wise SAE on pre-extracted reward-model activations."""

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        use_ddp: bool = True,
        local_rank: int = 0,
        start_epoch: int = 0,
        gpu_id: int = None,
        target_layer: int = None,
        learning_rate: float = 1e-3,
    ):
        self.d_model = d_model
        self.d_sae = d_sae
        self.use_ddp = use_ddp
        self.local_rank = local_rank
        self.start_epoch = start_epoch
        self.target_layer = target_layer
        self.learning_rate = learning_rate

        if torch.cuda.is_available():
            if use_ddp:
                self.device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(local_rank)
            elif gpu_id is not None:
                self.device = torch.device(f"cuda:{gpu_id}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.sae = StandardSAE(d_model, d_sae).to(self.device)

        if self.use_ddp and dist.is_initialized():
            self.sae = DDP(
                self.sae, device_ids=[local_rank], output_device=local_rank
            )

        self.optimizer = optim.Adam(self.sae.parameters(), lr=self.learning_rate)

        self.train_history = {
            "reconstruction_loss": [],
            "sparsity_loss": [],
            "total_loss": [],
            "l0_norm": [],
            "l1_norm": [],
        }

    def compute_loss(
        self, x: torch.Tensor, l1_coefficient: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        x_hat, z = self.sae(x)
        reconstruction_loss = torch.mean((x - x_hat) ** 2)
        sparsity_loss = torch.mean(torch.abs(z))
        total_loss = reconstruction_loss + l1_coefficient * sparsity_loss
        l0_norm = torch.mean((z > 0).float().sum(dim=1))
        l1_norm = torch.mean(torch.sum(torch.abs(z), dim=1))
        return {
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
            "total_loss": total_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
        }

    def train_step(
        self, x: torch.Tensor, l1_coefficient: float = 1e-3
    ) -> Dict[str, float]:
        self.optimizer.zero_grad()
        losses = self.compute_loss(x, l1_coefficient)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.sae._normalize_decoder()
        return {k: v.item() for k, v in losses.items()}

    def validate_step(
        self, batch_data: torch.Tensor, l1_coefficient: float
    ) -> Dict[str, float]:
        with torch.no_grad():
            x_hat, z = self.sae(batch_data)
            reconstruction_loss = nn.MSELoss()(x_hat, batch_data)
            sparsity_loss = torch.mean(torch.abs(z))
            total_loss = reconstruction_loss + l1_coefficient * sparsity_loss
            l0_norm = torch.mean((z > 1e-6).float().sum(dim=1))
            l1_norm = torch.mean(torch.abs(z).sum(dim=1))
            return {
                "reconstruction_loss": reconstruction_loss.item(),
                "sparsity_loss": sparsity_loss.item(),
                "total_loss": total_loss.item(),
                "l0_norm": l0_norm.item(),
                "l1_norm": l1_norm.item(),
            }

    def train(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 2048,
        l1_coefficient: float = 1e-3,
        save_interval: int = 10,
        checkpoint_dir: str = "checkpoints",
    ):
        is_main = (
            not self.use_ddp or not dist.is_initialized() or dist.get_rank() == 0
        )

        if checkpoint_dir is None:
            checkpoint_dir = (
                f"checkpoints/layer_{self.target_layer}"
                if self.target_layer is not None
                else "checkpoints"
            )

        if is_main:
            os.makedirs(checkpoint_dir, exist_ok=True)

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)

        if self.use_ddp and dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        for epoch in range(num_epochs):
            current_epoch = self.start_epoch + epoch

            if self.use_ddp and dist.is_initialized():
                train_sampler.set_epoch(current_epoch)

            epoch_losses = {k: [] for k in self.train_history}
            self.sae.train()

            pbar = (
                tqdm(train_loader, desc=f"Epoch {current_epoch + 1}")
                if is_main
                else train_loader
            )

            for (batch_data,) in pbar:
                batch_data = batch_data.to(self.device)
                losses = self.train_step(batch_data, l1_coefficient)
                for k, v in losses.items():
                    epoch_losses[k].append(v)

            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

            self.sae.eval()
            val_losses = {k: [] for k in self.train_history}
            with torch.no_grad():
                for (batch_data,) in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_val = self.validate_step(batch_data, l1_coefficient)
                    for k, v in batch_val.items():
                        val_losses[k].append(v)
            avg_val = {k: np.mean(v) for k, v in val_losses.items()}

            if self.use_ddp and dist.is_initialized():
                for k in avg_losses:
                    t = torch.tensor(avg_losses[k], device=self.device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    avg_losses[k] = (t / dist.get_world_size()).item()
                for k in avg_val:
                    t = torch.tensor(avg_val[k], device=self.device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    avg_val[k] = (t / dist.get_world_size()).item()

            if is_main:
                for k, v in avg_losses.items():
                    self.train_history[k].append(v)

                print(
                    f"Epoch {current_epoch + 1}/{self.start_epoch + num_epochs}: "
                    f"Train Recon={avg_losses['reconstruction_loss']:.6f}, "
                    f"Sparsity={avg_losses['sparsity_loss']:.6f}, "
                    f"L0={avg_losses['l0_norm']:.1f} | "
                    f"Val Recon={avg_val['reconstruction_loss']:.6f}"
                )

                if (current_epoch + 1) % save_interval == 0:
                    path = os.path.join(
                        checkpoint_dir,
                        f"sae_checkpoint_epoch_{current_epoch + 1}.pt",
                    )
                    self.save_checkpoint(path, current_epoch + 1)

        if is_main:
            final_path = os.path.join(checkpoint_dir, "sae_final_model.pt")
            self.save_checkpoint(final_path, self.start_epoch + num_epochs)

    def save_checkpoint(self, path: str, epoch: int):
        if self.use_ddp and dist.is_initialized() and dist.get_rank() != 0:
            return
        model_state = (
            self.sae.module.state_dict()
            if hasattr(self.sae, "module")
            else self.sae.state_dict()
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "d_model": self.d_model,
                "d_sae": self.d_sae,
                "train_history": self.train_history,
                "start_epoch": self.start_epoch,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        if hasattr(self.sae, "module"):
            self.sae.module.load_state_dict(ckpt["model_state_dict"])
        else:
            self.sae.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.train_history = ckpt.get("train_history", self.train_history)
        return ckpt.get("epoch", 0)

    def plot_training_curves(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].plot(self.train_history["reconstruction_loss"])
        axes[0, 0].set_title("Reconstruction Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.train_history["sparsity_loss"])
        axes[0, 1].set_title("Sparsity Loss (L1)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].grid(True)

        axes[1, 0].plot(self.train_history["l0_norm"])
        axes[1, 0].set_title("L0 Norm (Active Features)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.train_history["total_loss"])
        axes[1, 1].set_title("Total Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def process_single_data(data: dict) -> List[str]:
    """Concatenate question and output texts from a single data record."""
    texts = []
    question = data.get("question", "")
    if not question:
        return texts
    for output_item in data.get("output", []):
        if isinstance(output_item, dict):
            output_text = output_item.get("text", "")
            if output_text:
                texts.append(f"{question} {output_text}".strip())
    return texts


def load_training_texts(data_dir: str) -> List[str]:
    """Load reasoning-path texts from JSONL files under *data_dir*."""
    texts = []
    if not os.path.exists(data_dir):
        return texts
    json_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".jsonl"):
                json_files.append(os.path.join(root, f))

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            texts.extend(process_single_data(data))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            continue
    return texts


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp(rank: int, world_size: int, backend: str = "nccl", master_port: str = "12355"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAE Training on PRM activations")
    parser.add_argument("--model-path", type=str, required=True, help="Path to PRM model")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with JSONL reasoning paths")
    parser.add_argument("--target-layer", type=int, required=True, help="Transformer block layer index")
    parser.add_argument("--d-model", type=int, required=True, help="Hidden size of the PRM")
    parser.add_argument("--expansion-factor", type=int, default=8, help="SAE expansion factor m = d_model * factor")
    parser.add_argument("--l1-coefficient", type=float, default=1e-3, help="Sparsity coefficient alpha")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Training data ratio")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max sequence length")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id for single-GPU training")
    args = parser.parse_args()

    d_sae = args.d_model * args.expansion_factor

    texts = load_training_texts(args.data_dir)
    if len(texts) < 100:
        print("Insufficient training texts.")
        return

    cache_path = f"activations_layer_{args.target_layer}_cache.pt"
    dataset = ActivationDataset(
        args.model_path, args.target_layer, use_ddp=False, gpu_id=args.gpu
    )
    activations = dataset.load_activations(cache_path)
    if activations is None:
        activations = dataset.extract_activations_from_texts(texts, max_length=args.max_length)
        if len(activations) == 0:
            print("Failed to extract activations.")
            return
        dataset.save_activations(activations, cache_path)

    num_train = int(args.train_ratio * len(activations))
    train_data = activations[:num_train]
    val_data = activations[num_train:]

    trainer = SAETrainer(
        args.d_model,
        d_sae,
        use_ddp=False,
        start_epoch=0,
        gpu_id=args.gpu,
        target_layer=args.target_layer,
        learning_rate=args.learning_rate,
    )

    if args.resume and os.path.exists(args.resume):
        loaded_epoch = trainer.load_checkpoint(args.resume)
        trainer.start_epoch = loaded_epoch

    trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        l1_coefficient=args.l1_coefficient,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.plot_training_curves(f"sae_training_curves_{timestamp}.png")


if __name__ == "__main__":
    main()
