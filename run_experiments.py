#!/usr/bin/env python3
"""
Run experiments for CS336 Assignment 1.

This script provides utilities for:
1. Learning rate tuning experiments
2. Batch size experiments
3. Ablation studies
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    name: str
    base_args: dict[str, Any]
    sweep_param: str | None = None
    sweep_values: list[Any] | None = None


def run_training(args: dict[str, Any], experiment_dir: str) -> str:
    """Run training with given arguments.

    Args:
        args: Training arguments
        experiment_dir: Directory to save results

    Returns:
        Path to checkpoint
    """
    # Build command
    cmd = [sys.executable, "train.py"]

    for key, value in args.items():
        # Use the key as-is (argparse uses snake_case by default)
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # Set checkpoint directory
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    cmd.extend(["--checkpoint_dir", checkpoint_dir])

    # Set wandb run name
    if args.get("use_wandb", False):
        run_name = args.get("wandb_run_name") or Path(experiment_dir).name
        cmd.extend(["--wandb_run_name", run_name])

    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 60)

    # Run training
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"Training failed with code {result.returncode}")

    return checkpoint_dir


def run_lr_sweep(
    base_args: dict[str, Any],
    learning_rates: list[float],
    experiment_dir: str,
    use_wandb: bool = False,
) -> dict[str, Any]:
    """Run learning rate sweep experiments.

    Args:
        base_args: Base training arguments
        learning_rates: List of learning rates to try
        experiment_dir: Directory to save results
        use_wandb: Whether to use wandb

    Returns:
        Dictionary of results
    """
    results = {}

    for lr in learning_rates:
        print(f"\n{'=' * 60}")
        print(f"Learning Rate: {lr}")
        print("=" * 60)

        # Create experiment directory
        lr_str = f"lr_{lr:.0e}".replace("-", "m")
        run_dir = os.path.join(experiment_dir, lr_str)
        os.makedirs(run_dir, exist_ok=True)

        # Update args
        args = base_args.copy()
        args["learning_rate"] = lr
        args["use_wandb"] = use_wandb

        # Run training
        try:
            checkpoint_dir = run_training(args, run_dir)

            # Load final checkpoint to get loss
            import torch

            checkpoint_path = os.path.join(checkpoint_dir, "final.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                results[str(lr)] = {
                    "learning_rate": lr,
                    "checkpoint": checkpoint_path,
                    "status": "completed",
                }
            else:
                results[str(lr)] = {"learning_rate": lr, "status": "no_checkpoint"}
        except Exception as e:
            print(f"Error: {e}")
            results[str(lr)] = {
                "learning_rate": lr,
                "status": "failed",
                "error": str(e),
            }

    # Save results
    results_path = os.path.join(experiment_dir, "lr_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLR Sweep Results saved to {results_path}")
    return results


def run_batch_size_sweep(
    base_args: dict[str, Any],
    batch_sizes: list[int],
    experiment_dir: str,
    use_wandb: bool = False,
) -> dict[str, Any]:
    """Run batch size sweep experiments.

    Args:
        base_args: Base training arguments
        batch_sizes: List of batch sizes to try
        experiment_dir: Directory to save results
        use_wandb: Whether to use wandb

    Returns:
        Dictionary of results
    """
    results = {}

    for batch_size in batch_sizes:
        print(f"\n{'=' * 60}")
        print(f"Batch Size: {batch_size}")
        print("=" * 60)

        # Create experiment directory
        run_dir = os.path.join(experiment_dir, f"batch_{batch_size}")
        os.makedirs(run_dir, exist_ok=True)

        # Update args
        args = base_args.copy()
        args["batch_size"] = batch_size
        args["use_wandb"] = use_wandb

        # Run training
        try:
            checkpoint_dir = run_training(args, run_dir)

            checkpoint_path = os.path.join(checkpoint_dir, "final.pt")
            if os.path.exists(checkpoint_path):
                results[str(batch_size)] = {
                    "batch_size": batch_size,
                    "checkpoint": checkpoint_path,
                    "status": "completed",
                }
            else:
                results[str(batch_size)] = {
                    "batch_size": batch_size,
                    "status": "no_checkpoint",
                }
        except Exception as e:
            print(f"Error: {e}")
            results[str(batch_size)] = {
                "batch_size": batch_size,
                "status": "failed",
                "error": str(e),
            }

    # Save results
    results_path = os.path.join(experiment_dir, "batch_size_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBatch Size Sweep Results saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run experiments")

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["lr_sweep", "batch_sweep", "single"],
        required=True,
        help="Type of experiment",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments",
        help="Experiment output directory",
    )

    # Data
    parser.add_argument("--train_data", type=str, default="data/train.npy")
    parser.add_argument("--valid_data", type=str, default="data/valid.npy")

    # Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Sweep parameters
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3, 3e-3],
        help="Learning rates for lr_sweep",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Batch sizes for batch_sweep",
    )

    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.experiment_dir, f"{args.experiment}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Base arguments
    base_args = {
        "train_data": args.train_data,
        "valid_data": args.valid_data,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "batch_size": args.batch_size,
        "total_steps": args.total_steps,
        "learning_rate": args.learning_rate,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "device": args.device,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
    }

    # Save config
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Run experiment
    if args.experiment == "lr_sweep":
        results = run_lr_sweep(
            base_args, args.learning_rates, experiment_dir, args.use_wandb
        )
    elif args.experiment == "batch_sweep":
        results = run_batch_size_sweep(
            base_args, args.batch_sizes, experiment_dir, args.use_wandb
        )
    elif args.experiment == "single":
        run_dir = os.path.join(experiment_dir, "single_run")
        os.makedirs(run_dir, exist_ok=True)
        run_training(base_args, run_dir)

    print("\n" + "=" * 60)
    print("Experiments complete!")
    print(f"Results saved to {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
