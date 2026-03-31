from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import regex
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .common import gpt2_bytes_to_unicode


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    # y = W @ x^T, where W is [d_out, d_in] and x is [..., d_in]
    # Result: [..., d_out]
    return torch.matmul(in_features, weights.T)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    # Lookup embedding for each token id
    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # FFN(x) = W_2(SiLU(W_1 x) ⊙ W_3 x)
    # W1: [d_ff, d_model], W2: [d_model, d_ff], W3: [d_ff, d_model]

    # W_1 x: [..., d_ff]
    w1_out = torch.matmul(in_features, w1_weight.T)

    # W_3 x: [..., d_ff]
    w3_out = torch.matmul(in_features, w3_weight.T)

    # SiLU(W_1 x) ⊙ W_3 x
    # SiLU(x) = x * sigmoid(x)
    silu_out = w1_out * torch.sigmoid(w1_out)
    hidden = silu_out * w3_out

    # W_2(hidden): [..., d_model]
    output = torch.matmul(hidden, w2_weight.T)

    return output


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T
    # Q: [..., queries, d_k], K: [..., keys, d_k]
    # scores: [..., queries, keys]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)

    # Apply mask if provided
    # mask: True = can attend, False = cannot attend
    if mask is not None:
        # Set masked positions to -inf
        scores = scores.masked_fill(~mask, float("-inf"))

    # Apply softmax
    attn_weights = run_softmax(scores, dim=-1)

    # Multiply by values
    # attn_weights: [..., queries, keys], V: [..., keys, d_v]
    # output: [..., queries, d_v]
    output = torch.matmul(attn_weights, V)

    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # d_k = d_v = d_model // num_heads
    d_k = d_model // num_heads

    # Project Q, K, V in single matrix multiply each
    # in_features: [..., seq_len, d_model]
    # q_proj_weight: [d_model, d_model] (d_k * num_heads = d_model)
    # Q: [..., seq_len, d_model]
    Q = torch.matmul(in_features, q_proj_weight.T)
    K = torch.matmul(in_features, k_proj_weight.T)
    V = torch.matmul(in_features, v_proj_weight.T)

    # Reshape for multi-head attention
    # in_features: [..., seq_len, d_model] where ... is batch dimensions
    # We want: [..., num_heads, seq_len, d_k]
    batch_dims = in_features.shape[:-2]  # All dimensions except seq_len and d_model
    seq_len = in_features.shape[-2]

    # [..., seq_len, d_model] -> [..., seq_len, num_heads, d_k] -> [..., num_heads, seq_len, d_k]
    Q = Q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)

    # Create causal mask (self-attention)
    # Each position can attend to itself and all previous positions
    # mask: [seq_len, seq_len] where mask[i, j] = True if j <= i
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
    )

    # Apply scaled dot product attention
    # The mask will be broadcast to match the batch and head dimensions
    attn_output = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    # Reshape back: [..., num_heads, seq_len, d_k] -> [..., seq_len, d_model]
    attn_output = attn_output.transpose(-3, -2).contiguous()
    attn_output = attn_output.view(*batch_dims, seq_len, d_model)

    # Output projection
    output = torch.matmul(attn_output, o_proj_weight.T)

    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # d_k = d_v = d_model // num_heads
    d_k = d_model // num_heads

    # Project Q, K, V in single matrix multiply each
    # in_features: [..., seq_len, d_model]
    # Q: [..., seq_len, d_model]
    Q = torch.matmul(in_features, q_proj_weight.T)
    K = torch.matmul(in_features, k_proj_weight.T)
    V = torch.matmul(in_features, v_proj_weight.T)

    # Reshape for multi-head attention
    # in_features: [..., seq_len, d_model] where ... is batch dimensions
    # We want: [..., num_heads, seq_len, d_k]
    batch_dims = in_features.shape[:-2]  # All dimensions except seq_len and d_model
    seq_len = in_features.shape[-2]

    # [..., seq_len, d_model] -> [..., seq_len, num_heads, d_k] -> [..., num_heads, seq_len, d_k]
    Q = Q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)

    # Apply RoPE to Q and K
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device)
        # Expand for batch dimensions
        token_positions = token_positions.expand(*batch_dims, seq_len)

    # Apply RoPE to Q and K
    # Q: [..., num_heads, seq_len, d_k]
    # We need to apply RoPE per head, so reshape temporarily
    original_shape = Q.shape
    # Flatten batch and head dimensions for RoPE: [..., num_heads, seq_len, d_k] -> [-1, seq_len, d_k]
    Q_flat = Q.reshape(-1, seq_len, d_k)
    K_flat = K.reshape(-1, seq_len, d_k)

    # token_positions: [..., seq_len] needs to be expanded for num_heads
    # [..., seq_len] -> [..., 1, seq_len] -> [..., num_heads, seq_len] -> [-1, seq_len]
    pos_expanded = token_positions.unsqueeze(-2).expand(*batch_dims, num_heads, seq_len)
    pos_flat = pos_expanded.reshape(-1, seq_len)

    Q_rope = run_rope(d_k, theta, max_seq_len, Q_flat, pos_flat)
    K_rope = run_rope(d_k, theta, max_seq_len, K_flat, pos_flat)

    # Reshape back
    Q = Q_rope.view(*original_shape)
    K = K_rope.view(*original_shape)

    # Create causal mask (self-attention)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
    )

    # Apply scaled dot product attention
    attn_output = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    # Reshape back: [..., num_heads, seq_len, d_k] -> [..., seq_len, d_model]
    attn_output = attn_output.transpose(-3, -2).contiguous()
    attn_output = attn_output.view(*batch_dims, seq_len, d_model)

    # Output projection
    output = torch.matmul(attn_output, o_proj_weight.T)

    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    # Precompute sin/cos cache for positions 0 to max_seq_len-1
    # Shape: [max_seq_len, d_k/2]
    half_dim = d_k // 2
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, d_k, 2, dtype=torch.float32, device=in_query_or_key.device)
            / d_k
        )
    )
    positions = torch.arange(
        max_seq_len, dtype=torch.float32, device=in_query_or_key.device
    )
    # Shape: [max_seq_len, d_k/2]
    angles = torch.outer(positions, freqs)

    cos_cache = torch.cos(angles)  # [max_seq_len, d_k/2]
    sin_cache = torch.sin(angles)  # [max_seq_len, d_k/2]

    # Get sin/cos for the given token_positions
    # token_positions: [..., sequence_length]
    # Need to index into cos_cache and sin_cache
    original_shape = token_positions.shape
    pos_flat = token_positions.flatten()  # Flatten batch dimensions

    cos_pos = cos_cache[pos_flat]  # [prod(batch), d_k/2]
    sin_pos = sin_cache[pos_flat]  # [prod(batch), d_k/2]

    # Reshape back to [..., sequence_length, d_k/2]
    batch_shape = original_shape
    cos_pos = cos_pos.view(*batch_shape, half_dim)
    sin_pos = sin_pos.view(*batch_shape, half_dim)

    # Split input into even and odd dimensions
    x = in_query_or_key.float()
    x1 = x[..., 0::2]  # [..., sequence_length, d_k/2] - even indices
    x2 = x[..., 1::2]  # [..., sequence_length, d_k/2] - odd indices

    # Apply rotation
    # [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated_x1 = x1 * cos_pos - x2 * sin_pos
    rotated_x2 = x1 * sin_pos + x2 * cos_pos

    # Interleave back
    output = torch.stack([rotated_x1, rotated_x2], dim=-1)
    output = output.flatten(-2)  # [..., sequence_length, d_k]

    return output.to(in_query_or_key.dtype)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    # Pre-norm Transformer block:
    # y = x + MultiHeadSelfAttention(RMSNorm(x))
    # z = y + SwiGLU(RMSNorm(y))

    seq_len = in_features.shape[1]

    # First sublayer: Attention with residual
    # RMSNorm
    normed1 = run_rmsnorm(d_model, 1e-5, weights["ln1.weight"], in_features)

    # Create token positions
    token_positions = torch.arange(seq_len, device=in_features.device).expand(
        in_features.shape[0], seq_len
    )

    # Multi-head self-attention with RoPE
    attn_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=normed1,
        token_positions=token_positions,
    )

    # Residual connection
    y = in_features + attn_output

    # Second sublayer: FFN with residual
    # RMSNorm
    normed2 = run_rmsnorm(d_model, 1e-5, weights["ln2.weight"], y)

    # SwiGLU FFN
    ffn_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=normed2,
    )

    # Residual connection
    z = y + ffn_output

    return z


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    # Get embeddings
    # token_embeddings.weight: [vocab_size, d_model]
    # in_indices: [batch_size, seq_len]
    # x: [batch_size, seq_len, d_model]
    x = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices,
    )

    # Process through each transformer layer
    for layer_idx in range(num_layers):
        # Build weights dict for this layer
        layer_weights = {}
        prefix = f"layers.{layer_idx}."
        for key in weights:
            if key.startswith(prefix):
                layer_weights[key.replace(prefix, "")] = weights[key]

        # Apply transformer block
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )

    # Final RMSNorm
    x = run_rmsnorm(d_model, 1e-5, weights["ln_final.weight"], x)

    # Output projection (lm_head)
    # lm_head.weight: [vocab_size, d_model]
    # We need: x @ lm_head.weight.T -> [batch_size, seq_len, vocab_size]
    # But run_linear expects weights as [d_out, d_in]
    # lm_head.weight is [vocab_size, d_model] which is [d_out, d_in]
    logits = run_linear(
        d_in=d_model,
        d_out=vocab_size,
        weights=weights["lm_head.weight"],
        in_features=x,
    )

    return logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    # Promote to float32 for numerical stability
    original_dtype = in_features.dtype
    x = in_features.float()

    # RMS(x) = sqrt(1/d_model * sum(x_i^2) + eps)
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

    # Normalize and scale
    output = (x / rms) * weights.float()

    # Return to original dtype
    return output.to(original_dtype)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    # SiLU(x) = x * sigmoid(x)
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Maximum starting index: need to have context_length + 1 tokens (input + next token)
    max_start_idx = len(dataset) - context_length - 1

    if max_start_idx < 0:
        raise ValueError(
            f"Dataset too small for context_length {context_length}: "
            f"need at least {context_length + 1} tokens, got {len(dataset)}"
        )

    # Random starting indices for each batch element
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Build x and y sequences with pre-allocated arrays (memory-efficient for memmap)
    # x: [batch_size, context_length] starting from start_indices
    # y: [batch_size, context_length] starting from start_indices + 1
    x = np.empty((batch_size, context_length), dtype=dataset.dtype)
    y = np.empty((batch_size, context_length), dtype=dataset.dtype)
    for i, idx in enumerate(start_indices):
        x[i] = dataset[idx : idx + context_length]
        y[i] = dataset[idx + 1 : idx + context_length + 1]

    # Convert to tensors and move to device
    return torch.from_numpy(x).to(device=device, dtype=torch.long), torch.from_numpy(
        y
    ).to(device=device, dtype=torch.long)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # Numerically stable softmax: subtract max for stability
    max_val = torch.max(in_features, dim=dim, keepdim=True).values
    shifted = in_features - max_val
    exp_shifted = torch.exp(shifted)
    sum_exp = torch.sum(exp_shifted, dim=dim, keepdim=True)
    return exp_shifted / sum_exp


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Numerically stable cross-entropy:
    # ℓ_i = -log(softmax(o_i)[x_{i+1}])
    # = -log(exp(o_i[target]) / sum_j exp(o_j))
    # = -o_i[target] + log(sum_j exp(o_j))

    # For numerical stability, subtract max from logits
    # log(sum_j exp(o_j - max)) + max = log(sum_j exp(o_j))
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted_logits = inputs - max_logits

    # log(sum_j exp(shifted_logits))
    log_sum_exp = shifted_logits.exp().sum(dim=-1).log()

    # Get the logit for the target class using gather (handles arbitrary batch dims)
    target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(
        -1
    )

    # Cross-entropy: -target_logits + log_sum_exp
    loss_per_example = -target_logits + log_sum_exp

    # Return average loss
    return loss_per_example.mean()


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Collect all gradients into a single vector and compute total L2 norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.data.pow(2).sum()
    total_norm = total_norm.sqrt().item()

    # Clip if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """

    class AdamW(torch.optim.Optimizer):
        """AdamW optimizer implementation following the paper's Algorithm 2."""

        def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
        ):
            if lr < 0:
                raise ValueError(f"Invalid learning rate: {lr}")
            if eps < 0:
                raise ValueError(f"Invalid epsilon value: {eps}")
            if betas[0] < 0 or betas[1] < 0:
                raise ValueError(f"Invalid beta values: {betas}")
            if weight_decay < 0:
                raise ValueError(f"Invalid weight_decay value: {weight_decay}")

            defaults = {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
            super().__init__(params, defaults)

        def step(self, closure=None):
            """Perform a single optimization step."""
            loss = None if closure is None else closure()

            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["t"] = 0
                        state["m"] = torch.zeros_like(p.data)
                        state["v"] = torch.zeros_like(p.data)

                    state["t"] += 1
                    t = state["t"]
                    m = state["m"]
                    v = state["v"]

                    # Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # Update biased second raw moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Compute bias-corrected estimates: m̂_t = m_t / (1 - β₁^t), v̂_t = v_t / (1 - β₂^t)
                    bias_correction1 = 1 - beta1**t
                    bias_correction2 = 1 - beta2**t
                    step_size = lr / bias_correction1
                    denom = v.sqrt().add_(eps) / math.sqrt(bias_correction2)

                    # Parameter update with weight decay (AdamW style):
                    # θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε) - α * λ * θ_{t-1}
                    # Weight decay must use the ORIGINAL parameter value
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.addcdiv_(m, denom, value=-step_size)

            return loss

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    import math

    # Phase 1: Linear warmup
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    # Phase 2: Cosine annealing
    if it < cosine_cycle_iters:
        # Progress through the cosine cycle (0 to 1)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # Cosine decay from max to min
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * progress)
        )

    # Phase 3: Return minimum learning rate after cosine cycle
    return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """

    class Tokenizer:
        """BPE Tokenizer implementation."""

        # GPT-2 pretokenization pattern
        GPT2_PATTERN = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
        ):
            self.vocab = vocab
            self.merges = merges

            # Build reverse vocab: bytes -> id
            self.vocab_reverse: dict[bytes, int] = {v: k for k, v in vocab.items()}

            # Build merge rankings (lower = higher priority)
            self.merge_ranks: dict[tuple[bytes, bytes], int] = {
                merge: i for i, merge in enumerate(merges)
            }

            # Handle special tokens
            self.special_tokens: list[str] = special_tokens or []
            # Build regex pattern for splitting on special tokens
            # Sort by length descending to match longer tokens first (handles overlapping tokens)
            if self.special_tokens:
                sorted_special_tokens = sorted(
                    self.special_tokens, key=len, reverse=True
                )
                pattern = "|".join(map(regex.escape, sorted_special_tokens))
                self.special_token_pattern = regex.compile(f"({pattern})")
            else:
                self.special_token_pattern = None

        @classmethod
        def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None,
        ):
            """Construct a Tokenizer from serialized vocabulary and merges files.

            Args:
                vocab_filepath: Path to the vocabulary JSON file (GPT-2 format)
                merges_filepath: Path to the merges text file
                special_tokens: Optional list of special tokens

            Returns:
                A Tokenizer instance
            """
            # Use GPT-2 byte mapping from common.py
            byte_encoder = gpt2_bytes_to_unicode()
            byte_decoder = {v: k for k, v in byte_encoder.items()}

            # Load vocabulary from JSON file
            with open(vocab_filepath, encoding="utf-8") as f:
                gpt2_vocab = json.load(f)

            # Convert GPT-2 vocab (unicode strings) to bytes
            vocab: dict[int, bytes] = {}
            for token_str, token_id in gpt2_vocab.items():
                token_bytes = bytes([byte_decoder[ch] for ch in token_str])
                vocab[token_id] = token_bytes

            # Add special tokens if they don't exist in vocab
            if special_tokens:
                existing_bytes = set(vocab.values())
                for special_token in special_tokens:
                    special_bytes = special_token.encode("utf-8")
                    if special_bytes not in existing_bytes:
                        vocab[len(vocab)] = special_bytes

            # Load merges from text file
            merges: list[tuple[bytes, bytes]] = []
            with open(merges_filepath, encoding="utf-8") as f:
                for line in f:
                    cleaned_line = line.rstrip()
                    if cleaned_line and len(cleaned_line.split(" ")) == 2:
                        token1, token2 = cleaned_line.split(" ")
                        token1_bytes = bytes([byte_decoder[ch] for ch in token1])
                        token2_bytes = bytes([byte_decoder[ch] for ch in token2])
                        merges.append((token1_bytes, token2_bytes))

            return cls(vocab, merges, special_tokens)

        def _pretokenize(self, text: str) -> list[str]:
            """Split text into tokens using GPT-2 regex pattern."""
            return self.GPT2_PATTERN.findall(text)

        def _split_on_special_tokens(self, text: str) -> list[str]:
            """Split text on special token boundaries."""
            if self.special_token_pattern is None:
                return [text] if text else []

            parts = self.special_token_pattern.split(text)
            return [p for p in parts if p]

        def _apply_bpe(self, token_bytes: list[bytes]) -> list[bytes]:
            """Apply BPE merges to a list of byte tokens."""
            if len(token_bytes) <= 1:
                return token_bytes

            while True:
                # Find the pair with the lowest merge rank (highest priority)
                best_pair = None
                best_rank = float("inf")

                for i in range(len(token_bytes) - 1):
                    pair = (token_bytes[i], token_bytes[i + 1])
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair

                if best_pair is None:
                    break

                # Merge all occurrences of the best pair
                new_tokens = []
                i = 0
                while i < len(token_bytes):
                    if (
                        i < len(token_bytes) - 1
                        and (token_bytes[i], token_bytes[i + 1]) == best_pair
                    ):
                        new_tokens.append(token_bytes[i] + token_bytes[i + 1])
                        i += 2
                    else:
                        new_tokens.append(token_bytes[i])
                        i += 1
                token_bytes = new_tokens

                if len(token_bytes) <= 1:
                    break

            return token_bytes

        def encode(self, text: str) -> list[int]:
            """Encode text to token IDs."""
            if not text:
                return []

            all_ids: list[int] = []

            # Split on special tokens first
            parts = self._split_on_special_tokens(text)

            for part in parts:
                if not part:
                    continue

                # Check if this part is a special token
                if part in self.special_tokens:
                    # Encode special token directly
                    part_bytes = part.encode("utf-8")
                    if part_bytes in self.vocab_reverse:
                        all_ids.append(self.vocab_reverse[part_bytes])
                    continue

                # Pretokenize and process each word
                for word in self._pretokenize(part):
                    # Convert word to list of single-byte tokens
                    token_bytes = [bytes([b]) for b in word.encode("utf-8")]

                    # Apply BPE
                    merged_tokens = self._apply_bpe(token_bytes)

                    # Convert to IDs
                    for token in merged_tokens:
                        if token in self.vocab_reverse:
                            all_ids.append(self.vocab_reverse[token])

            return all_ids

        def decode(self, ids: list[int]) -> str:
            """Decode token IDs to text."""
            byte_sequence = b""
            for token_id in ids:
                if token_id in self.vocab:
                    byte_sequence += self.vocab[token_id]
            return byte_sequence.decode("utf-8", errors="replace")

        def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
            """Encode an iterable of strings to token IDs (memory efficient)."""
            for text in iterable:
                yield from self.encode(text)

    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # GPT-2 pretokenization pattern
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def pretokenize(text: str) -> list[str]:
        """Split text into tokens using GPT-2 regex pattern."""
        return GPT2_PATTERN.findall(text)

    def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
        """Split text on special token boundaries."""
        # Build regex pattern to match any special token
        pattern = "|".join(map(regex.escape, special_tokens))
        # Split and keep delimiters
        parts = regex.split(f"({pattern})", text)
        return [p for p in parts if p]

    # Step 1: Initialize vocabulary with 256 byte values
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Add special tokens to vocab
    special_token_set = set(special_tokens)
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # Step 2: Read and preprocess the training text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on special tokens and pretokenize each part
    # Use word frequencies: unique words with counts
    word_freqs: dict[tuple[bytes, ...], int] = Counter()

    parts = split_on_special_tokens(text, special_tokens)
    for part in parts:
        if part in special_token_set:
            continue
        elif part:
            for word in pretokenize(part):
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                word_freqs[word_bytes] += 1

    merges: list[tuple[bytes, bytes]] = []

    # Build initial pair counts with frequencies
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq

    # Step 3: BPE training loop
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # Select highest frequency pair (with lexicographically larger as tiebreaker)
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

        if pair_counts[best_pair] == 0:
            break

        # Create new token by merging
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        # Remove best_pair from counts
        del pair_counts[best_pair]

        # Process all words - merge and update counts incrementally
        new_word_freqs: dict[tuple[bytes, ...], int] = {}
        for word, freq in word_freqs.items():
            # Convert to list for merging
            tokens = list(word)
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == best_pair:
                    # Remove old pairs around this position
                    if i > 0:
                        left_pair = (tokens[i - 1], tokens[i])
                        pair_counts[left_pair] -= freq
                        if pair_counts[left_pair] <= 0:
                            pair_counts.pop(left_pair, None)
                    if i + 2 < len(tokens):
                        right_pair = (tokens[i + 1], tokens[i + 2])
                        pair_counts[right_pair] -= freq
                        if pair_counts[right_pair] <= 0:
                            pair_counts.pop(right_pair, None)

                    # Perform merge
                    tokens[i : i + 2] = [new_token]

                    # Add new pairs around the merged token
                    if i > 0:
                        left_pair = (tokens[i - 1], new_token)
                        pair_counts[left_pair] += freq
                    if i + 1 < len(tokens):
                        right_pair = (new_token, tokens[i + 1])
                        pair_counts[right_pair] += freq
                    # Don't increment i - check if new token can merge again
                else:
                    i += 1

            new_word_freqs[tuple(tokens)] = freq

        word_freqs = new_word_freqs

    return vocab, merges
