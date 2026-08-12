"""Minimal K64 Q/K quantizer used by the released SM89 mixed path.

This is the two-function subset of SpargeAttention's Apache-2.0 Triton
quantizer that the K64 executor actually consumes.  Keeping it local avoids
importing or building Sparge's unrelated sparse-attention extensions.

Copyright (c) 2025 SpargeAttention team. Apache License 2.0.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_quantize_kernel(
    x_ptr,
    x_mean_ptr,
    output_ptr,
    scale_ptr,
    TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)
    heads = tl.num_programs(1)
    blocks = tl.num_programs(2)

    rows = block_id * BLOCK + tl.arange(0, BLOCK)[:, None]
    columns = tl.arange(0, HEAD_DIM)[None, :]
    offsets = (
        batch * heads * TOKENS * HEAD_DIM
        + head * TOKENS * HEAD_DIM
        + rows * HEAD_DIM
        + columns
    )
    valid = rows < TOKENS
    values = tl.load(x_ptr + offsets, mask=valid, other=0.0)
    if SUBTRACT_MEAN:
        mean = tl.load(
            x_mean_ptr
            + batch * heads * HEAD_DIM
            + head * HEAD_DIM
            + columns
        )
        values -= mean
        values = tl.where(valid, values, 0.0)

    values_fp32 = values.to(tl.float32)
    scale = tl.max(tl.abs(values_fp32)) / 127.0 + 1.0e-7
    quantized = values_fp32 / scale
    quantized += 0.5 * tl.where(quantized >= 0, 1, -1)
    tl.store(output_ptr + offsets, quantized.to(tl.int8), mask=valid)
    tl.store(
        scale_ptr + batch * heads * blocks + head * blocks + block_id,
        scale,
    )


def _quantize(
    tensor: torch.Tensor,
    mean: torch.Tensor | None,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = tensor.contiguous()
    batch, heads, tokens, head_dim = tensor.shape
    blocks = (tokens + block - 1) // block
    output = torch.empty_like(tensor, dtype=torch.int8)
    scales = torch.empty(
        (batch, heads, blocks), device=tensor.device, dtype=torch.float32
    )
    _qk_quantize_kernel[(batch, heads, blocks)](
        tensor,
        mean,
        output,
        scales,
        TOKENS=tokens,
        HEAD_DIM=head_dim,
        BLOCK=block,
        SUBTRACT_MEAN=mean is not None,
    )
    return output, scales


def quantize_qk_k64(
    query: torch.Tensor,
    key: torch.Tensor,
    key_mean: torch.Tensor | None = None,
    query_block: int = 64,
    key_block: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed-INT8 Q/K and one FP32 scale per physical block."""

    q8, q_scale = _quantize(query, None, query_block)
    k8, k_scale = _quantize(key, key_mean, key_block)
    return q8, q_scale, k8, k_scale


__all__ = ["quantize_qk_k64"]
