"""Native SM89 executor for the production stripe-compact ragged route."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from evg.layers.attention.mpa.backends.sm89_k64 import (
    assemble_h3_k64_output,
    native_k64_mixed_attention,
    pack_h3_k64_qkv,
    pool_compact_k64_key,
)
from evg.layers.attention.mpa.layout import materialize_ragged_2d_layout
from evg.layers.attention.mpa.routing import (
    draft_probability,
    route_probability,
)


_BLOCK = 64


def _warmup_sync(stage: str, device: torch.device) -> None:
    if os.environ.get("EVG_MPA_WARMUP_SYNC") != "1":
        return
    try:
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(f"SM89 warm-up failed after {stage}") from exc


def _pool_query(query_fp16: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, head_dim = query_fp16.shape
    blocks = tokens // _BLOCK
    denominator = counts.to(torch.float32).view(batch, 1, blocks, 1)
    return (
        query_fp16.view(batch, heads, blocks, _BLOCK, head_dim)
        .sum(dim=3, dtype=torch.float32)
        .div_(denominator)
        .to(torch.float16)
        .contiguous()
    )


def _dense_prefix_sdpa(
    query_bshd: torch.Tensor,
    key_bshd: torch.Tensor,
    value_bshd: torch.Tensor,
) -> torch.Tensor:
    """Match Sol-H3's original-dtype dense prefix-query overwrite."""

    return F.scaled_dot_product_attention(
        query_bshd.permute(0, 2, 1, 3),
        key_bshd.permute(0, 2, 1, 3),
        value_bshd.permute(0, 2, 1, 3),
        dropout_p=0.0,
        is_causal=False,
    )


def sm89_ragged_h3_attention(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    *,
    prefix_tokens: int,
    sparsity_ratio: float,
    retained_fp8_ratio: float,
    retained_fp16_ratio: float,
    video_shape: tuple[int, int, int],
    diag_jensen: bool = False,
) -> torch.Tensor:
    """Run ragged 2-D routing through the optimized native Q64xK64 kernel."""

    if q_bshd.shape != k_bshd.shape or q_bshd.shape != v_bshd.shape:
        raise ValueError("Q/K/V must share [1,S,H,128]")
    if (
        q_bshd.ndim != 4
        or q_bshd.size(0) != 1
        or q_bshd.size(-1) != 128
        or q_bshd.dtype not in (torch.float16, torch.bfloat16)
        or k_bshd.dtype != q_bshd.dtype
        or v_bshd.dtype != q_bshd.dtype
        or q_bshd.device.type != "cuda"
        or k_bshd.device != q_bshd.device
        or v_bshd.device != q_bshd.device
    ):
        raise ValueError("Q/K/V must be same-dtype CUDA [1,S,H,128]")
    if torch.cuda.get_device_capability(q_bshd.device) != (8, 9):
        raise RuntimeError("the released native path requires SM89")
    if type(prefix_tokens) is not int or not 0 < prefix_tokens < q_bshd.size(1):
        raise ValueError("prefix_tokens must split the packed sequence")
    if (
        not isinstance(video_shape, tuple)
        or len(video_shape) != 3
        or any(type(value) is not int or value <= 0 for value in video_shape)
        or math.prod(video_shape) != q_bshd.size(1) - prefix_tokens
    ):
        raise ValueError("video_shape must match the post-prefix token count")
    if type(diag_jensen) is not bool:
        raise TypeError("diag_jensen must be bool")
    ratios = (float(retained_fp8_ratio), float(retained_fp16_ratio))
    if (
        any(not math.isfinite(value) or value < 0.0 for value in ratios)
        or ratios[0] <= 0.0
        or ratios[1] <= 0.0
        or not math.isclose(sum(ratios), 1.0, abs_tol=1.0e-6)
    ):
        raise ValueError("FP8/FP16 ratios must be positive and sum to one")

    frames, height, width = video_shape
    prefix_output = _dense_prefix_sdpa(q_bshd[:, :prefix_tokens], k_bshd, v_bshd)
    layout = materialize_ragged_2d_layout(
        q_bshd.device,
        frames=frames,
        height=height,
        width=width,
        logical_block=_BLOCK,
    )
    q_packed, key_fp16, value_fp16 = pack_h3_k64_qkv(
        q_bshd.permute(0, 2, 1, 3),
        k_bshd.permute(0, 2, 1, 3),
        v_bshd.permute(0, 2, 1, 3),
        layout.indices,
        layout.slot_valid,
        prefix_tokens,
    )
    _warmup_sync("QKV packing", q_bshd.device)
    batch, heads, _, _ = q_packed.shape
    video_blocks = layout.counts.numel()
    video_counts = layout.counts.view(1, video_blocks).expand(batch, video_blocks).contiguous()
    prefix_blocks = math.ceil(prefix_tokens / _BLOCK)
    prefix_capacity = prefix_blocks * _BLOCK
    prefix_counts = (
        torch.clamp(
            prefix_tokens
            - torch.arange(prefix_blocks, device=q_bshd.device, dtype=torch.int32) * _BLOCK,
            min=0,
            max=_BLOCK,
        )
        .view(1, prefix_blocks)
        .expand(batch, prefix_blocks)
        .contiguous()
    )
    valid_k = torch.cat((prefix_counts, video_counts), dim=1).contiguous()
    key_video = key_fp16[:, :, prefix_capacity:]

    q_pool = _pool_query(q_packed, video_counts)
    k_pool = pool_compact_k64_key(key_video, video_counts)
    moments = diag_jensen
    q_second = None
    k_second = None
    if moments:
        denominator = video_counts.to(torch.float32).view(1, 1, video_blocks, 1)
        q_second = (
            q_packed.view(1, heads, video_blocks, _BLOCK, -1)
            .float()
            .square_()
            .sum(dim=3)
            .div_(denominator)
            .to(torch.float16)
            .contiguous()
        )
        k_second = (
            key_video.view(1, heads, video_blocks, _BLOCK, -1)
            .float()
            .square_()
            .sum(dim=3)
            .div_(denominator)
            .to(torch.float16)
            .contiguous()
        )
    probability = draft_probability(
        q_pool,
        k_pool,
        q_second=q_second,
        k_second=k_second,
    )
    route = route_probability(
        probability,
        layout.anchors,
        anchor_count=layout.anchor_count,
        prefix_blocks=prefix_blocks,
        sparsity_ratio=float(sparsity_ratio),
        fp8_ratio=ratios[0],
        fp16_ratio=ratios[1],
    )
    _warmup_sync("DraftMap routing", q_bshd.device)

    key_blocks = prefix_blocks + video_blocks
    ids = torch.zeros(
        (batch, heads, video_blocks, key_blocks),
        device=q_bshd.device,
        dtype=torch.int32,
    )
    ids[..., :video_blocks] = route.block_ids
    fp8_counts = route.fp8_counts.contiguous()
    fp16_counts = (route.fp16_counts + prefix_blocks).contiguous()

    del prefix_counts, video_counts
    del key_video, q_pool, k_pool, q_second, k_second, probability, route
    video_output, _ = native_k64_mixed_attention(
        q_packed,
        key_fp16,
        value_fp16,
        ids,
        fp8_counts,
        ids,
        fp16_counts,
        valid_k,
        fp16_prefix_blocks=prefix_blocks,
    )
    _warmup_sync("mixed attention", q_bshd.device)
    del q_packed, key_fp16, value_fp16, valid_k, ids
    output = assemble_h3_k64_output(prefix_output, video_output, layout.inverse)
    _warmup_sync("output assembly", q_bshd.device)
    return output


__all__ = ["sm89_ragged_h3_attention"]
