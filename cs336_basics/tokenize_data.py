#!/usr/bin/env python3
"""
Data preprocessing script for CS336 Assignment 1.

This script:
1. Trains a BPE tokenizer on the training data
2. Tokenizes training and validation data
3. Saves tokenized data as numpy arrays
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tests.adapters import get_tokenizer, run_train_bpe
from tests.common import gpt2_bytes_to_unicode


def train_tokenizer(
    train_path: str,
    vocab_size: int,
    special_tokens: list[str],
    output_dir: str,
    num_workers: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE tokenizer and save vocabulary and merges.

    Args:
        train_path: Path to training text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens
        output_dir: Directory to save vocab and merges

    Returns:
        Tuple of (vocab, merges)
    """
    print(f"Training BPE tokenizer on {train_path}...")
    print(f"  Target vocab size: {vocab_size}")
    print(f"  Special tokens: {special_tokens}")

    # Count lines for progress bar
    with open(train_path, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    print(f"  Total lines in training file: {num_lines:,}")

    start_time = time.time()

    vocab, merges = run_train_bpe(
        train_path, vocab_size, special_tokens, num_workers=num_workers
    )

    elapsed = time.time() - start_time
    print(f"  Final vocab size: {len(vocab)}")
    print(f"  Number of merges: {len(merges)}")
    print(f"  Training time: {elapsed / 60:.1f} minutes")

    # Save vocab
    vocab_path = os.path.join(output_dir, "vocab.json")

    # Use GPT-2 byte mapping to convert bytes to readable strings
    byte_decoder = gpt2_bytes_to_unicode()

    def bytes_to_readable(b: bytes) -> str:
        """Convert bytes to readable string using GPT-2 byte mapping."""
        return "".join(byte_decoder[byte] for byte in b)

    vocab_json = {str(k): bytes_to_readable(v) for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    print(f"  Saved vocab to {vocab_path}")

    # Save merges
    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for pair in merges:
            f.write(f"{bytes_to_readable(pair[0])} {bytes_to_readable(pair[1])}\n")
    print(f"  Saved merges to {merges_path}")

    return vocab, merges


def tokenize_file(
    input_path: str,
    output_path: str,
    tokenizer,
    dtype: np.dtype = np.uint16,
) -> int:
    """Tokenize a text file and save as numpy array.

    Args:
        input_path: Path to input text file
        output_path: Path to output numpy file
        tokenizer: BPE tokenizer instance
        dtype: Data type for tokens

    Returns:
        Total number of tokens
    """
    print(f"Tokenizing {input_path}...")

    # Count lines for progress bar
    with open(input_path, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    print(f"  Total lines: {num_lines:,}")

    start_time = time.time()

    # Read and tokenize with progress bar
    all_tokens = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="  Tokenizing", unit="lines"):
            tokens = tokenizer.encode(line)
            all_tokens.extend(tokens)

    elapsed = time.time() - start_time
    print(f"  Total: {len(all_tokens):,} tokens")
    print(f"  Tokenization time: {elapsed / 60:.1f} minutes")
    print(f"  Average speed: {len(all_tokens) / elapsed:,.0f} tokens/sec")

    # Convert to numpy array and save
    token_array = np.array(all_tokens, dtype=dtype)
    np.save(output_path, token_array)
    print(f"  Saved to {output_path}")

    return len(all_tokens)


def main():
    parser = argparse.ArgumentParser(description="Preprocess training data")

    parser.add_argument(
        "--train_text", type=str, required=True, help="Path to training text file"
    )
    parser.add_argument(
        "--valid_text", type=str, required=True, help="Path to validation text file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=10000, help="Target vocabulary size"
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for pretokenization (default: 1 for sequential)",
    )
    parser.add_argument(
        "--skip_tokenizer_training",
        action="store_true",
        help="Skip tokenizer training (use existing vocab/merges)",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Path to existing vocab.json (if skip_tokenizer_training)",
    )
    parser.add_argument(
        "--merges_path",
        type=str,
        default=None,
        help="Path to existing merges.txt (if skip_tokenizer_training)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train or load tokenizer
    if args.skip_tokenizer_training:
        if not args.vocab_path or not args.merges_path:
            raise ValueError(
                "Must provide vocab_path and merges_path if skip_tokenizer_training"
            )

        print(f"Loading tokenizer from {args.vocab_path} and {args.merges_path}...")

        # Use GPT-2 byte mapping to convert strings back to bytes
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        def readable_to_bytes(s: str) -> bytes:
            """Convert readable string back to bytes using GPT-2 byte mapping."""
            return bytes([byte_decoder[ch] for ch in s])

        with open(args.vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {int(k): readable_to_bytes(v) for k, v in vocab_json.items()}

        merges = []
        with open(args.merges_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append(
                        (readable_to_bytes(parts[0]), readable_to_bytes(parts[1]))
                    )
    else:
        vocab, merges = train_tokenizer(
            args.train_text,
            args.vocab_size,
            args.special_tokens,
            args.output_dir,
            args.num_workers,
        )

    # Create tokenizer
    print(f"  Vocab size: {len(vocab)}, Merges: {len(merges)}")
    print("  Creating tokenizer...")
    create_start = time.time()
    tokenizer = get_tokenizer(vocab, merges, args.special_tokens)
    print(f"  Tokenizer created in {time.time() - create_start:.2f}s")

    # Tokenize training data
    train_output = os.path.join(args.output_dir, "train.npy")
    tokenize_file(args.train_text, train_output, tokenizer)

    # Tokenize validation data
    valid_output = os.path.join(args.output_dir, "valid.npy")
    tokenize_file(args.valid_text, valid_output, tokenizer)

    print("\nData preprocessing complete!")
    print(f"  Training data: {train_output}")
    print(f"  Validation data: {valid_output}")


if __name__ == "__main__":
    main()
