#!/usr/bin/env python3
"""
Training script for CS336 Assignment 1 - Transformer Language Model.

This script trains a transformer language model on text data using:
- BPE tokenization
- Cosine learning rate schedule with warmup
- AdamW optimizer
- Gradient clipping
- Checkpoint saving
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

# Import from adapters
from tests.adapters import (
    get_adamw_cls,
    run_cross_entropy,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_load_checkpoint,
    run_save_checkpoint,
)

# =============================================================================
# Model Definition
# =============================================================================


def init_linear_weight(d_in: int, d_out: int) -> torch.Tensor:
    """Initialize linear layer weights with truncated normal distribution.

    N(0, 2/(d_in + d_out)) truncated at ±3σ
    """
    std = math.sqrt(2.0 / (d_in + d_out))
    weight = torch.empty(d_out, d_in)
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return weight


def init_embedding_weight(vocab_size: int, d_model: int) -> torch.Tensor:
    """Initialize embedding weights with truncated normal distribution.

    N(0, 1) truncated at ±3
    """
    weight = torch.empty(vocab_size, d_model)
    nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    return weight


class TransformerLM(nn.Module):
    """Transformer Language Model implemented from scratch."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # Token embeddings
        self.token_embeddings = nn.Parameter(init_embedding_weight(vocab_size, d_model))

        # Transformer layers - use plain dicts, register parameters with the module
        self.layers = []

        for _ in range(num_layers):
            layer = {
                # Attention projections
                "attn": {
                    "q_proj": init_linear_weight(d_model, d_model),
                    "k_proj": init_linear_weight(d_model, d_model),
                    "v_proj": init_linear_weight(d_model, d_model),
                    "output_proj": init_linear_weight(d_model, d_model),
                },
                # Layer norms
                "ln1": torch.ones(d_model),
                "ln2": torch.ones(d_model),
                # FFN
                "ffn": {
                    "w1": init_linear_weight(d_model, d_ff),
                    "w2": init_linear_weight(d_ff, d_model),
                    "w3": init_linear_weight(d_model, d_ff),
                },
            }
            self.layers.append(layer)

        # Register layer parameters with the module (with layer index)
        for layer_idx, layer in enumerate(self.layers):
            for key in ["q_proj", "k_proj", "v_proj", "output_proj"]:
                self.register_parameter(
                    f"attn_{key}_{layer_idx}", nn.Parameter(layer["attn"][key])
                )
            self.register_parameter(f"ln1_{layer_idx}", nn.Parameter(layer["ln1"]))
            self.register_parameter(f"ln2_{layer_idx}", nn.Parameter(layer["ln2"]))
            for key in ["w1", "w2", "w3"]:
                self.register_parameter(
                    f"ffn_{key}_{layer_idx}", nn.Parameter(layer["ffn"][key])
                )

        # Final layer norm
        self.ln_final = nn.Parameter(torch.ones(d_model))

        # Output projection (lm_head)
        self.lm_head = nn.Parameter(init_embedding_weight(vocab_size, d_model))

        # Pre-build weights cache for each layer (avoid rebuilding in forward)
        self._layer_weights_cache = []
        for layer_idx in range(num_layers):
            weights = {
                "attn.q_proj.weight": getattr(self, f"attn_q_proj_{layer_idx}"),
                "attn.k_proj.weight": getattr(self, f"attn_k_proj_{layer_idx}"),
                "attn.v_proj.weight": getattr(self, f"attn_v_proj_{layer_idx}"),
                "attn.output_proj.weight": getattr(
                    self, f"attn_output_proj_{layer_idx}"
                ),
                "ln1.weight": getattr(self, f"ln1_{layer_idx}"),
                "ln2.weight": getattr(self, f"ln2_{layer_idx}"),
                "ffn.w1.weight": getattr(self, f"ffn_w1_{layer_idx}"),
                "ffn.w2.weight": getattr(self, f"ffn_w2_{layer_idx}"),
                "ffn.w3.weight": getattr(self, f"ffn_w3_{layer_idx}"),
            }
            self._layer_weights_cache.append(weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [batch_size, seq_len] token IDs

        Returns:
            [batch_size, seq_len, vocab_size] logits
        """
        batch_size, seq_len = input_ids.shape

        # Truncate if needed
        if seq_len > self.context_length:
            input_ids = input_ids[:, -self.context_length :]
            seq_len = self.context_length

        # Token embeddings
        x = self.token_embeddings[input_ids]  # [batch, seq, d_model]

        # Build weights dict for each layer and apply transformer block
        from tests.adapters import run_transformer_block

        for layer_idx in range(self.num_layers):
            # Use cached weights for this layer
            weights = self._layer_weights_cache[layer_idx]

            x = run_transformer_block(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                max_seq_len=self.context_length,
                theta=self.rope_theta,
                weights=weights,
                in_features=x,
            )

        # Final layer norm
        from tests.adapters import run_rmsnorm

        x = run_rmsnorm(self.d_model, 1e-5, self.ln_final, x)

        # Output projection
        logits = torch.matmul(x, self.lm_head.T)

        return logits


# =============================================================================
# Data Loading
# =============================================================================


def load_dataset(data_path: str, dtype: np.dtype = np.uint16) -> np.ndarray:
    """Load tokenized dataset using memory mapping.

    Args:
        data_path: Path to .npy or .bin file containing token IDs
        dtype: Data type for the array

    Returns:
        Memory-mapped numpy array
    """
    if data_path.endswith(".npy"):
        return np.load(data_path, mmap_mode="r")
    elif data_path.endswith(".bin"):
        # Raw binary file
        return np.memmap(data_path, dtype=dtype, mode="r")
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


# =============================================================================
# Training Utilities
# =============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 10,
) -> float:
    """Estimate loss on dataset."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = run_get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            loss = run_cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()

    model.train()
    return total_loss / num_batches


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return math.exp(loss)


# =============================================================================
# Text Generation
# =============================================================================


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample next token from logits.

    Args:
        logits: [batch_size, vocab_size] logits
        temperature: Temperature for softmax
        top_p: Nucleus sampling threshold

    Returns:
        [batch_size] sampled token IDs
    """
    # Greedy decoding when temperature is 0
    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    # Apply temperature
    logits = logits / temperature

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Apply top-p (nucleus) sampling
    if top_p < 1.0:
        # Sort probabilities descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff index
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Zero out removed probabilities
        sorted_probs[sorted_indices_to_remove] = 0.0

        # Renormalize
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample from sorted distribution
        sampled_sorted_indices = torch.multinomial(sorted_probs, num_samples=1).squeeze(
            -1
        )

        # Map back to original indices
        sampled_tokens = sorted_indices.gather(
            -1, sampled_sorted_indices.unsqueeze(-1)
        ).squeeze(-1)
    else:
        # Sample from full distribution
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return sampled_tokens


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """Generate text tokens from prompt.

    Args:
        model: Transformer language model
        prompt_ids: List of prompt token IDs
        max_new_tokens: Maximum number of new tokens to generate
        context_length: Maximum context length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        eos_token_id: End-of-sequence token ID (stop generation if encountered)
        device: Device to run on

    Returns:
        List of generated token IDs (including prompt)
    """
    model.eval()

    # Convert prompt to tensor
    ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        # Truncate to context length
        context = ids[-context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        # Get logits for last position
        logits = model(x)
        next_token_logits = logits[0, -1, :]

        # Sample next token
        next_token = sample_next_token(
            next_token_logits.unsqueeze(0), temperature, top_p
        )
        next_token = next_token.item()

        ids.append(next_token)

        # Stop if EOS token
        if eos_token_id is not None and next_token == eos_token_id:
            break

    model.train()
    return ids


# =============================================================================
# Training Loop
# =============================================================================


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    train_data_path: str = "data/train.npy"
    valid_data_path: str = "data/valid.npy"

    # Model architecture
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0

    # Training
    batch_size: int = 64
    total_steps: int = 5000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # AdamW
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    resume_from: str | None = None

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "cs336-assignment1"
    wandb_run_name: str | None = None

    def to_dict(self) -> dict:
        """Return model config dict for serialization (HuggingFace style)."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
        }


def train(config: TrainingConfig) -> None:
    """Main training function."""

    print("=" * 60)
    print("CS336 Assignment 1 - Transformer Language Model Training")
    print("=" * 60)

    # Print configuration
    print(f"\nConfiguration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print()

    # Initialize wandb
    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Set device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_data = load_dataset(config.train_data_path)
    valid_data = load_dataset(config.valid_data_path)
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Valid tokens: {len(valid_data):,}")

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Save model config (HuggingFace style - save once at start)
    config_path = os.path.join(config.checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved config to {config_path}")

    # Create model
    print("\nCreating model...")
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )
    model.to(device)

    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    # Create optimizer
    AdamW = get_adamw_cls()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )

    # Load checkpoint if resuming
    start_step = 0
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"\nLoading checkpoint from {config.resume_from}...")
        start_step = run_load_checkpoint(config.resume_from, model, optimizer)
        print(f"  Resuming from step {start_step}")

    # Training loop
    print("\nStarting training...")
    print(f"  Total steps: {config.total_steps}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Context length: {config.context_length}")
    print()

    # Calculate total tokens
    total_tokens = config.batch_size * config.total_steps * config.context_length
    print(f"  Total tokens to process: {total_tokens:,}")
    print()

    # Training metrics
    best_val_loss = float("inf")
    train_losses = []

    start_time = time.time()
    iter_time = time.time()

    for step in range(start_step, config.total_steps):
        # Get learning rate
        lr = run_get_lr_cosine_schedule(
            step,
            config.learning_rate,
            config.min_learning_rate,
            config.warmup_steps,
            config.total_steps,
        )

        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        x, y = run_get_batch(
            train_data, config.batch_size, config.context_length, config.device
        )

        # Forward pass
        logits = model(x)

        # Compute loss
        loss = run_cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        run_gradient_clipping(model.parameters(), config.grad_clip)

        # Optimizer step
        optimizer.step()

        # Record loss
        train_losses.append(loss.item())

        # Logging
        if (step + 1) % config.log_interval == 0:
            elapsed = time.time() - iter_time
            tokens_per_sec = (
                config.batch_size
                * config.context_length
                * config.log_interval
                / elapsed
            )
            avg_loss = sum(train_losses[-config.log_interval :]) / config.log_interval
            ppl = compute_perplexity(avg_loss)

            print(
                f"Step {step + 1}/{config.total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL: {ppl:.2f} | "
                f"LR: {lr:.2e} | "
                f"Tokens/sec: {tokens_per_sec:,.0f}"
            )

            if config.use_wandb:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/perplexity": ppl,
                        "train/learning_rate": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step": step + 1,
                    }
                )

            iter_time = time.time()

        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            val_loss = estimate_loss(
                model,
                valid_data,
                config.batch_size,
                config.context_length,
                config.device,
                config.eval_batches,
            )
            val_ppl = compute_perplexity(val_loss)

            print(f"\n  Validation Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}\n")

            if config.use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                        "train/step": step + 1,
                    }
                )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(config.checkpoint_dir, "best.pt")
                run_save_checkpoint(model, optimizer, step + 1, checkpoint_path)
                print(f"  Saved best checkpoint to {checkpoint_path}")

        # Save periodic checkpoint
        if (step + 1) % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"step_{step + 1}.pt")
            run_save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(config.checkpoint_dir, "final.pt")
    run_save_checkpoint(model, optimizer, config.total_steps, final_checkpoint_path)
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")

    # Final evaluation
    final_val_loss = estimate_loss(
        model,
        valid_data,
        config.batch_size,
        config.context_length,
        config.device,
        config.eval_batches * 5,
    )
    final_val_ppl = compute_perplexity(final_val_loss)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Final Validation Loss: {final_val_loss:.4f}")
    print(f"  Final Validation Perplexity: {final_val_ppl:.2f}")
    print(f"  Total Training Time: {total_time / 3600:.2f} hours")
    print(f"  Total Steps: {config.total_steps}")
    print(f"  Total Tokens: {total_tokens:,}")
    print()

    if config.use_wandb:
        wandb.log(
            {
                "val/final_loss": final_val_loss,
                "val/final_perplexity": final_val_ppl,
            }
        )
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")

    # Data
    parser.add_argument("--train_data", type=str, default="data/train.npy")
    parser.add_argument("--valid_data", type=str, default="data/valid.npy")

    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # AdamW
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    config = TrainingConfig(
        train_data_path=args.train_data,
        valid_data_path=args.valid_data,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    train(config)


if __name__ == "__main__":
    main()
