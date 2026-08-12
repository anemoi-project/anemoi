"""DraftMap probability and fixed-budget spatial-cross routing for SM89."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import NamedTuple

import torch

from .build_identity import import_native_extension, resolve_mixed_attention_operator


_INT32_MAX = 2**31 - 1


class RoutePlan(NamedTuple):
    """Native route tensors consumed by the K64 mixed-attention wrapper."""

    precision_map: torch.Tensor
    packed_ids: torch.Tensor
    low8_counts: torch.Tensor
    fp16_counts: torch.Tensor
    low4_counts: torch.Tensor
    first_empty_row_sentinel: torch.Tensor
    fp16_blocks_per_head: int
    low8_blocks_per_head: int
    low4_blocks_per_head: int


class _RouterOps(NamedTuple):
    draft: object
    route_spatial_cross: object


@lru_cache(maxsize=1)
def _load_router_ops() -> _RouterOps:
    import_native_extension("router")
    return _RouterOps(
        draft=resolve_mixed_attention_operator("draft"),
        route_spatial_cross=resolve_mixed_attention_operator(
            "route_spatial_cross"
        ),
    )


def _route_counts(
    blocks_per_head: int,
    *,
    sparsity_ratio: float,
    retained_fp8_ratio: float,
    retained_fp16_ratio: float,
) -> tuple[int, int, int]:
    """Hamilton-allocate the retained budget as ``(FP16, FP8, zero)``."""

    if type(blocks_per_head) is not int or blocks_per_head <= 0:
        raise ValueError("blocks_per_head must be a positive integer")
    values = (
        sparsity_ratio,
        retained_fp16_ratio,
        retained_fp8_ratio,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise TypeError("sparsity and precision ratios must be real scalars")
    sparsity = float(sparsity_ratio)
    ratios = tuple(float(value) for value in values[1:])
    if not math.isfinite(sparsity) or not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity_ratio must be finite and in [0, 1)")
    if any(not math.isfinite(value) or value < 0.0 for value in ratios):
        raise ValueError("precision ratios must be finite and nonnegative")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("precision ratios must sum to one")
    ratios = tuple(value / total for value in ratios)

    keep = min(
        blocks_per_head,
        max(1, math.floor((1.0 - sparsity) * blocks_per_head + 0.5)),
    )
    quotas = tuple(keep * ratio for ratio in ratios)
    counts = [math.floor(quota) for quota in quotas]
    order = sorted(
        range(2),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[: keep - sum(counts)]:
        counts[index] += 1
    return counts[0], counts[1], 0


def _validate_pool(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        tensor.device.type != "cuda"
        or tensor.dtype != torch.float16
        or tensor.ndim != 4
        or not tensor.is_contiguous()
        or min(tensor.shape) <= 0
        or tensor.shape[-1] not in (64, 128)
    ):
        raise ValueError(
            f"{name} must be contiguous CUDA FP16 [B,H,blocks,64|128]"
        )


def compute_draft_probability_tensors(
    q_pool: torch.Tensor,
    k_pool: torch.Tensor,
) -> torch.Tensor:
    """Compute the FP16 row-softmax DraftMap from pooled Q and K."""

    _validate_pool(q_pool, "q_pool")
    _validate_pool(k_pool, "k_pool")
    if (
        q_pool.device != k_pool.device
        or q_pool.shape[0] != k_pool.shape[0]
        or q_pool.shape[-2:] != k_pool.shape[-2:]
        or q_pool.shape[1] % k_pool.shape[1]
    ):
        raise ValueError("q_pool and k_pool geometries are incompatible")
    with torch.cuda.device(q_pool.device):
        return _load_router_ops().draft(q_pool, k_pool)


def _native_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise RuntimeError(
            f"native {name} must be contiguous {dtype} {shape} on {device}"
        )
    return tensor


def route_draft_spatial_cross_precision_tensors(
    probability_fp16: torch.Tensor,
    sparsity_ratio: float,
    *,
    retained_low8_ratio: float,
    retained_fp16_ratio: float,
    frames: int,
    patches_h: int,
    patches_w: int,
) -> RoutePlan:
    """Global probability top-k with legal same-frame 2-D cross anchors.

    Anchors consume the unchanged retained budget and are promoted to FP16 by
    borrowing FP8 seats when necessary.
    """

    if (
        not isinstance(probability_fp16, torch.Tensor)
        or probability_fp16.device.type != "cuda"
        or probability_fp16.dtype != torch.float16
        or probability_fp16.ndim != 4
        or not probability_fp16.is_contiguous()
    ):
        raise ValueError(
            "probability_fp16 must be contiguous CUDA FP16 [B,H,R,R]"
        )
    batch, heads, patches, keys = probability_fp16.shape
    if min(batch, heads, patches) <= 0 or keys != patches:
        raise ValueError("probability_fp16 must be nonempty and square")
    if probability_fp16.numel() > _INT32_MAX:
        raise ValueError("routing item count exceeds int32")
    if any(type(value) is not int or value <= 0 for value in (frames, patches_h, patches_w)):
        raise ValueError("spatial-cross dimensions must be positive integers")
    if frames * patches_h * patches_w != patches:
        raise ValueError("spatial-cross geometry does not match DraftMap")

    n16, n8, n4 = _route_counts(
        patches * patches,
        sparsity_ratio=sparsity_ratio,
        retained_fp8_ratio=retained_low8_ratio,
        retained_fp16_ratio=retained_fp16_ratio,
    )
    anchors = frames * (
        patches_h * patches_w
        + 2
        * (
            patches_h * (patches_w - 1)
            + (patches_h - 1) * patches_w
        )
    )
    retained = n16 + n8 + n4
    if anchors > retained:
        raise ValueError("mandatory spatial-cross anchors exceed route budget")
    if anchors > n16:
        n16 = anchors
        n8 = retained - n16

    with torch.cuda.device(probability_fp16.device):
        outputs = _load_router_ops().route_spatial_cross(
            probability_fp16,
            n16,
            n8,
            n4,
            frames,
            patches_h,
            patches_w,
        )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError("native spatial-cross route must return six tensors")
    map_shape = (batch, heads, patches, patches)
    row_shape = (batch, heads, patches)
    device = probability_fp16.device
    return RoutePlan(
        _native_tensor(outputs[0], name="precision_map", shape=map_shape, dtype=torch.uint8, device=device),
        _native_tensor(outputs[1], name="packed_ids", shape=map_shape, dtype=torch.int32, device=device),
        _native_tensor(outputs[2], name="low8_counts", shape=row_shape, dtype=torch.int32, device=device),
        _native_tensor(outputs[3], name="fp16_counts", shape=row_shape, dtype=torch.int32, device=device),
        _native_tensor(outputs[4], name="low4_counts", shape=row_shape, dtype=torch.int32, device=device),
        _native_tensor(outputs[5], name="first_empty", shape=(1,), dtype=torch.int32, device=device),
        n16,
        n8,
        n4,
    )


__all__ = [
    "RoutePlan",
    "compute_draft_probability_tensors",
    "route_draft_spatial_cross_precision_tensors",
]
