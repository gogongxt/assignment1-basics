#!/usr/bin/env python3
"""
Text generation script for CS336 Assignment 1.

Generate text from a trained transformer language model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from tests.adapters import get_tokenizer
from tests.common import gpt2_bytes_to_unicode

# Import train module for TransformerLM and generate function
from train import TransformerLM
from train import generate as generate_tokens


def load_model(checkpoint_path: str, config_path: str, device: str = "cpu"):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config.json file
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Load config from config.json
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    model = TransformerLM(**config)
    num_layers = config["num_layers"]

    # Load state dict - keys match directly
    model.token_embeddings.data = state_dict["token_embeddings"]
    model.ln_final.data = state_dict["ln_final"]
    model.lm_head.data = state_dict["lm_head"]

    # Load layer parameters
    for layer_idx in range(num_layers):
        # Attention projections
        for key in ["q_proj", "k_proj", "v_proj", "output_proj"]:
            param_name = f"attn_{key}_{layer_idx}"
            if param_name in state_dict:
                getattr(model, param_name).data = state_dict[param_name]

        # Layer norms
        model.get_parameter(f"ln1_{layer_idx}").data = state_dict[f"ln1_{layer_idx}"]
        model.get_parameter(f"ln2_{layer_idx}").data = state_dict[f"ln2_{layer_idx}"]

        # FFN weights
        for key in ["w1", "w2", "w3"]:
            param_name = f"ffn_{key}_{layer_idx}"
            if param_name in state_dict:
                getattr(model, param_name).data = state_dict[param_name]

    model.to(device)
    model.eval()

    return model, config


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    context_length: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
    eos_token: str = "<|endoftext|>",
) -> str:
    """Generate text from prompt.

    Args:
        model: Transformer language model
        tokenizer: BPE tokenizer
        prompt: Text prompt
        max_new_tokens: Maximum number of new tokens to generate
        context_length: Maximum context length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        device: Device to run on
        eos_token: End-of-text token string

    Returns:
        Generated text
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt)

    # Get EOS token ID
    eos_token_id = None
    if eos_token:
        eos_token_ids = tokenizer.encode(eos_token)
        if eos_token_ids:
            eos_token_id = eos_token_ids[0]

    # Generate tokens
    output_ids = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device,
    )

    # Decode
    return tokenizer.decode(output_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to config.json"
    )
    parser.add_argument(
        "--vocab_path", type=str, default="data/vocab.json", help="Path to vocab.json"
    )
    parser.add_argument(
        "--merges_path", type=str, default="data/merges.txt", help="Path to merges.txt"
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens",
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time", help="Text prompt"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--context_length", type=int, default=256, help="Maximum context length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate"
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab_path} and {args.merges_path}...")

    # Use GPT-2 byte mapping to convert strings back to bytes
    byte_encoder = gpt2_bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    with open(args.vocab_path, "r") as f:
        vocab_json = json.load(f)
    vocab = {
        int(k): bytes([byte_decoder[token] for token in v])
        for k, v in vocab_json.items()
    }

    merges = []
    with open(args.merges_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                merges.append(
                    (
                        bytes([byte_decoder[token] for token in parts[0]]),
                        bytes([byte_decoder[token] for token in parts[1]]),
                    )
                )

    tokenizer = get_tokenizer(vocab, merges, args.special_tokens)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    print(f"Loading config from {args.config_path}...")
    model, config = load_model(args.checkpoint, args.config_path, args.device)
    print(f"Model config: {config}")

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)

    for i in range(args.num_samples):
        output = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            context_length=args.context_length,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )

        print(f"\n[Sample {i + 1}]")
        print(output)
        print("-" * 60)


if __name__ == "__main__":
    main()
