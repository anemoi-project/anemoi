"""Native SM89 Q64 x K64 executor for stripe-compact ragged routing."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch

from evg.layers.attention.mpa.build_identity import (
    import_native_extension,
    resolve_mixed_attention_operator,
)


_BLOCK = 64


def _warmup_sync(stage: str, device: torch.device) -> None:
    if os.environ.get("EVG_MPA_WARMUP_SYNC") != "1":
        return
    try:
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(f"SM89 warm-up failed after {stage}") from exc


@dataclass(frozen=True)
class _K64Ops:
    mixed_attention: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    preprocess_v_fp8: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    assemble_h3_output: Callable[..., torch.Tensor]
    pack_h3_qkv: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@lru_cache(maxsize=1)
def _load_k64_ops() -> _K64Ops:
    import_native_extension("attention")
    return _K64Ops(
        mixed_attention=resolve_mixed_attention_operator(
            "k64_mixed_attention_forward"
        ),
        preprocess_v_fp8=resolve_mixed_attention_operator("preprocess_v_fp8"),
        assemble_h3_output=resolve_mixed_attention_operator(
            "assemble_h3_k64_output"
        ),
        pack_h3_qkv=resolve_mixed_attention_operator("pack_h3_k64_qkv_fp16"),
    )


def prepare_k64_fp8_operands(
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Quantize packed Q64/K64 operands for the mixed SM89 executor.

    Q/K retain the inherited symmetric INT8 tensor-core representation; V is
    E4M3.  ``FP8 phase`` is the established public name for this lower-precision
    attention phase.  The setup boundary is intentionally separate from route
    construction so it can be fused into the H3 packer after the numerical
    contract is locked.
    """

    if query_fp16.dtype != torch.float16:
        raise TypeError("query_fp16 must be FP16")
    if key_fp16.dtype != torch.float16 or value_fp16.dtype != torch.float16:
        raise TypeError("key_fp16/value_fp16 must be FP16")
    if key_fp16.shape != value_fp16.shape:
        raise ValueError("key_fp16/value_fp16 shapes must match")
    if any(
        not tensor.is_contiguous()
        for tensor in (query_fp16, key_fp16, value_fp16)
    ):
        raise ValueError("packed K64 Q/K/V must be contiguous")
    if query_fp16.size(2) % _BLOCK or key_fp16.size(2) % _BLOCK:
        raise ValueError("packed Q/K token capacities must be multiples of 64")
    # Import lazily so the native K64 FP16 path does not depend on Triton.
    # The mixed path owns this minimal quantizer and therefore does not require
    # SpargeAttention's unrelated compiled baseline extensions.
    from evg.layers.attention.mpa.backends._sm89_qk_quant import quantize_qk_k64

    q8, q_scale, k8, k_scale = quantize_qk_k64(
        query_fp16,
        key_fp16,
        None,
        _BLOCK,
        _BLOCK,
    )
    _warmup_sync("Q/K quantization", query_fp16.device)
    v8, v_scale = _load_k64_ops().preprocess_v_fp8(value_fp16)
    _warmup_sync("V preprocessing", query_fp16.device)
    return q8, k8, v8, q_scale, k_scale, v_scale


def native_k64_mixed_attention(
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    fp8_block_ids: torch.Tensor,
    fp8_block_counts: torch.Tensor,
    fp16_block_ids: torch.Tensor,
    fp16_block_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
    *,
    fp16_prefix_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch Q64xK64 mixed attention with one compact absolute-stage list.

    ``fp8_block_ids`` and ``fp16_block_ids`` must be the same tensor. Each row
    stores FP8 video stages first and FP16 video stages second; exact prefix
    stages are implicit and counted only by ``fp16_prefix_blocks``.
    """

    if fp8_block_ids.data_ptr() != fp16_block_ids.data_ptr():
        raise ValueError(
            "native mixed K64 requires one aliased compact FP8/FP16 route list"
        )

    q8, k8, v8, q_scale, k_scale, v_scale = prepare_k64_fp8_operands(
        query_fp16, key_fp16, value_fp16
    )
    scale = 1.0 / math.sqrt(query_fp16.size(-1))
    return _load_k64_ops().mixed_attention(
        q8,
        k8,
        v8,
        query_fp16,
        key_fp16,
        value_fp16,
        fp8_block_ids,
        fp8_block_counts,
        fp16_block_ids,
        fp16_block_counts,
        q_scale,
        k_scale,
        v_scale,
        valid_k_counts,
        fp16_prefix_blocks,
        scale,
    )


def pack_h3_k64_qkv(
    query_bhtd: torch.Tensor,
    key_bhtd: torch.Tensor,
    value_bhtd: torch.Tensor,
    video_token_indices: torch.Tensor,
    video_slot_valid: torch.Tensor,
    prefix_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack exact prefix K/V and stripe-ragged video Q/K/V in one launch."""

    if query_bhtd.shape != key_bhtd.shape or query_bhtd.shape != value_bhtd.shape:
        raise ValueError("H3 K64 Q/K/V must share [B,H,T,D] shape")
    if query_bhtd.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("H3 K64 Q/K/V must be FP16 or BF16")
    if key_bhtd.dtype != query_bhtd.dtype or value_bhtd.dtype != query_bhtd.dtype:
        raise TypeError("H3 K64 Q/K/V dtypes must match")
    if type(prefix_tokens) is not int or not 0 < prefix_tokens < query_bhtd.size(2):
        raise ValueError("prefix_tokens must split the H3 sequence")
    return _load_k64_ops().pack_h3_qkv(
        query_bhtd,
        key_bhtd,
        value_bhtd,
        video_token_indices,
        video_slot_valid,
        prefix_tokens,
    )


def assemble_h3_k64_output(
    prefix_output_bhsd: torch.Tensor,
    video_output_bhsd_fp16: torch.Tensor,
    video_inverse_indices: torch.Tensor,
) -> torch.Tensor:
    """Fuse inverse-raster scatter, prefix append, dtype cast, and BSHD store."""

    if prefix_output_bhsd.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("H3 prefix output must be FP16 or BF16")
    if video_output_bhsd_fp16.dtype != torch.float16:
        raise TypeError("H3 packed video output must be FP16")
    return _load_k64_ops().assemble_h3_output(
        prefix_output_bhsd,
        video_output_bhsd_fp16,
        video_inverse_indices,
    )


def pool_compact_k64_key(
    key_fp16: torch.Tensor,
    valid_k_counts: torch.Tensor,
) -> torch.Tensor:
    """Return only FP16 K means when skipped-output compensation is disabled."""

    if key_fp16.ndim != 4 or key_fp16.dtype != torch.float16:
        raise TypeError("key must be a rank-4 FP16 tensor")
    if key_fp16.size(2) % _BLOCK:
        raise ValueError("physical K capacity must be a multiple of 64")
    batch, heads, tokens, head_dim = key_fp16.shape
    blocks = tokens // _BLOCK
    if valid_k_counts.shape != (batch, blocks):
        raise ValueError("valid_k_counts must have shape [B,Kblocks]")
    denominator = valid_k_counts.to(torch.float32).view(batch, 1, blocks, 1)
    return (
        key_fp16.view(batch, heads, blocks, _BLOCK, head_dim)
        .sum(dim=3, dtype=torch.float32)
        .div_(denominator)
        .to(torch.float16)
        .contiguous()
    )


__all__ = [
    "assemble_h3_k64_output",
    "native_k64_mixed_attention",
    "pack_h3_k64_qkv",
    "pool_compact_k64_key",
    "prepare_k64_fp8_operands",
]
