"""Adaptive regular-2D DraftMap attention for MiniMax-H3 on SM89."""

from __future__ import annotations

from functools import lru_cache
import math

import torch
import torch.nn.functional as F

from mpa.backends.sm89_k64 import (
    assemble_h3_k64_output,
    native_k64_mixed_attention,
    pack_h3_k64_qkv,
    pool_compact_k64_key,
)
from mpa.routing import (
    compute_draft_probability_tensors,
    route_draft_spatial_cross_precision_tensors,
)


_BLOCK = 64


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


@lru_cache(maxsize=32)
def _compact_raster2d_indices(
    device_index: int,
    frames: int,
    height: int,
    width: int,
    tile_height: int,
    tile_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack every logical 2-D tile into one independent physical K64 block."""

    if any(
        type(value) is not int or value <= 0
        for value in (frames, height, width, tile_height, tile_width)
    ):
        raise ValueError("video and tile dimensions must be positive integers")
    tile_tokens = tile_height * tile_width
    if tile_tokens > _BLOCK:
        raise ValueError("the SM89 demo requires logical tile area <= 64")

    device = torch.device("cuda", device_index)
    patches_h = math.ceil(height / tile_height)
    patches_w = math.ceil(width / tile_width)
    frame = torch.arange(frames, device=device, dtype=torch.int64).view(
        frames, 1, 1, 1, 1
    )
    patch_h = torch.arange(patches_h, device=device, dtype=torch.int64).view(
        1, patches_h, 1, 1, 1
    )
    patch_w = torch.arange(patches_w, device=device, dtype=torch.int64).view(
        1, 1, patches_w, 1, 1
    )
    local_h = torch.arange(tile_height, device=device, dtype=torch.int64).view(
        1, 1, 1, tile_height, 1
    )
    local_w = torch.arange(tile_width, device=device, dtype=torch.int64).view(
        1, 1, 1, 1, tile_width
    )
    raster_h = patch_h * tile_height + local_h
    raster_w = patch_w * tile_width + local_w
    valid = ((raster_h < height) & (raster_w < width)).expand(
        frames, patches_h, patches_w, tile_height, tile_width
    )
    raw = (
        frame * height * width + raster_h * width + raster_w
    ).expand(frames, patches_h, patches_w, tile_height, tile_width)

    logical = raw.reshape(-1, tile_tokens)
    logical_valid = valid.reshape(-1, tile_tokens)
    video_tokens = frames * height * width
    compact = torch.where(logical_valid, logical, video_tokens).sort(dim=-1).values
    counts = logical_valid.sum(dim=-1, dtype=torch.int32).contiguous()
    physical = torch.zeros(
        (compact.size(0), _BLOCK), device=device, dtype=torch.int64
    )
    slot = torch.arange(tile_tokens, device=device).view(1, tile_tokens)
    physical[:, :tile_tokens].copy_(
        torch.where(slot < counts.view(-1, 1), compact, torch.zeros_like(compact))
    )
    slot_valid = (
        torch.arange(_BLOCK, device=device).view(1, _BLOCK)
        < counts.view(-1, 1)
    ).contiguous()
    flat = physical.reshape(-1)
    flat_valid = slot_valid.reshape(-1)
    inverse = torch.empty(video_tokens, device=device, dtype=torch.int64)
    physical_slots = torch.arange(flat.numel(), device=device, dtype=torch.int64)
    inverse.scatter_(0, flat[flat_valid], physical_slots[flat_valid])
    return flat, flat_valid.contiguous(), counts, inverse


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


def sm89_regular2d_h3_attention(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    *,
    prefix_tokens: int,
    sparsity_ratio: float,
    retained_fp8_ratio: float,
    retained_fp16_ratio: float,
    video_shape: tuple[int, int, int],
    tile_shape: tuple[int, int],
) -> torch.Tensor:
    """Run the released FP8+FP16 K64 path and return contiguous BSHD output."""

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
    if (
        not isinstance(tile_shape, tuple)
        or len(tile_shape) != 2
        or any(type(value) is not int or value <= 0 for value in tile_shape)
        or math.prod(tile_shape) > _BLOCK
    ):
        raise ValueError("tile_shape must contain positive dimensions with area <= 64")
    ratios = (float(retained_fp8_ratio), float(retained_fp16_ratio))
    if (
        any(not math.isfinite(value) or value < 0.0 for value in ratios)
        or ratios[0] <= 0.0
        or ratios[1] <= 0.0
        or not math.isclose(sum(ratios), 1.0, abs_tol=1.0e-6)
    ):
        raise ValueError("FP8/FP16 ratios must be positive and sum to one")

    frames, height, width = video_shape
    tile_height, tile_width = tile_shape
    prefix_output = _dense_prefix_sdpa(
        q_bshd[:, :prefix_tokens], k_bshd, v_bshd
    )
    device_index = q_bshd.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    indices, slot_valid, video_counts_1d, inverse = _compact_raster2d_indices(
        device_index,
        frames,
        height,
        width,
        tile_height,
        tile_width,
    )
    q_packed, key_fp16, value_fp16 = pack_h3_k64_qkv(
        q_bshd.permute(0, 2, 1, 3),
        k_bshd.permute(0, 2, 1, 3),
        v_bshd.permute(0, 2, 1, 3),
        indices,
        slot_valid,
        prefix_tokens,
    )
    batch, heads, _, _ = q_packed.shape
    video_blocks = video_counts_1d.numel()
    video_counts = video_counts_1d.view(1, video_blocks).expand(
        batch, video_blocks
    ).contiguous()
    prefix_blocks = math.ceil(prefix_tokens / _BLOCK)
    prefix_capacity = prefix_blocks * _BLOCK
    prefix_counts = torch.clamp(
        prefix_tokens
        - torch.arange(prefix_blocks, device=q_bshd.device, dtype=torch.int32)
        * _BLOCK,
        min=0,
        max=_BLOCK,
    ).view(1, prefix_blocks).expand(batch, prefix_blocks).contiguous()
    valid_k = torch.cat((prefix_counts, video_counts), dim=1).contiguous()
    key_video = key_fp16[:, :, prefix_capacity:]

    q_pool = _pool_query(q_packed, video_counts)
    k_pool = pool_compact_k64_key(key_video, video_counts)
    probability = compute_draft_probability_tensors(q_pool, k_pool)
    route = route_draft_spatial_cross_precision_tensors(
        probability,
        sparsity_ratio=float(sparsity_ratio),
        retained_low8_ratio=ratios[0],
        retained_fp16_ratio=ratios[1],
        frames=frames,
        patches_h=math.ceil(height / tile_height),
        patches_w=math.ceil(width / tile_width),
    )

    key_blocks = prefix_blocks + video_blocks
    ids = torch.zeros(
        (batch, heads, video_blocks, key_blocks),
        device=q_bshd.device,
        dtype=torch.int32,
    )
    ids[..., :video_blocks] = route.packed_ids + prefix_blocks
    fp8_counts = route.low8_counts.contiguous()
    fp16_counts = (route.fp16_counts + prefix_blocks).contiguous()

    del indices, slot_valid, prefix_counts, video_counts
    del key_video, q_pool, k_pool, probability, route
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
    del q_packed, key_fp16, value_fp16, valid_k, ids
    return assemble_h3_k64_output(prefix_output, video_output, inverse)


__all__ = ["sm89_regular2d_h3_attention"]
