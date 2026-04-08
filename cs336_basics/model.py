#!/usr/bin/env python3
"""Transformer Language Model implementation from scratch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


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
