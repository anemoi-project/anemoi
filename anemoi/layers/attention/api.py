"""Stable Q/K/V attention entry point."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class VisualLayout:
    """Structured visual tokens following an optional packed prefix."""

    video_shape: tuple[int, int, int]
    prefix_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.video_shape, tuple)
            or len(self.video_shape) != 3
            or any(type(value) is not int or value <= 0 for value in self.video_shape)
        ):
            raise ValueError("video_shape must contain three positive integers")
        if type(self.prefix_tokens) is not int or self.prefix_tokens < 0:
            raise ValueError("prefix_tokens must be a nonnegative integer")

    @property
    def video_tokens(self) -> int:
        return math.prod(self.video_shape)

    @property
    def sequence_tokens(self) -> int:
        return self.prefix_tokens + self.video_tokens


@dataclass(frozen=True)
class SparseConfig:
    """Q geometry and zero-based, half-open per-layer sparsity schedule."""

    query_block_size: int = 64
    sparsity_ratio: float = 0.80
    layer_sparsity_bands: tuple[tuple[int, int, float], ...] = ()
    maxpool_weight: float = 0.0

    def __post_init__(self) -> None:
        if type(self.query_block_size) is not int or self.query_block_size not in (64, 128):
            raise ValueError("query_block_size must be 64 or 128")
        if (
            isinstance(self.sparsity_ratio, bool)
            or not isinstance(self.sparsity_ratio, (int, float))
            or not math.isfinite(float(self.sparsity_ratio))
            or not 0.0 <= float(self.sparsity_ratio) < 1.0
        ):
            raise ValueError("sparsity_ratio must be finite and in [0, 1)")
        if isinstance(self.maxpool_weight, bool) or not isinstance(
            self.maxpool_weight, (int, float)
        ):
            raise TypeError("maxpool_weight must be a real number")
        if not math.isfinite(float(self.maxpool_weight)) or not 0.0 <= float(
            self.maxpool_weight
        ) <= 1.0:
            raise ValueError("maxpool_weight must be finite and in [0, 1]")
        previous_end = 0
        for band in self.layer_sparsity_bands:
            if not isinstance(band, tuple) or len(band) != 3:
                raise TypeError("layer bands must be (first, last, sparsity)")
            first, last, ratio = band
            if (
                type(first) is not int
                or type(last) is not int
                or not 0 <= first < last
                or first < previous_end
            ):
                raise ValueError("layer bands must be sorted, disjoint, and nonnegative")
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(float(ratio))
                or not 0.0 <= float(ratio) < 1.0
            ):
                raise ValueError("layer sparsities must be finite and in [0, 1)")
            previous_end = last


@dataclass(frozen=True)
class QuantConfig:
    """Stable retained-block precision and independent prefix precision."""

    nvfp4_ratio: float = 0.0
    int8_ratio: float = 1.0
    fp16_ratio: float = 0.0
    prefix_kv_precision: str = "int8"
    prefix_query_precision: str = "int8"

    def __post_init__(self) -> None:
        ratios = (self.nvfp4_ratio, self.int8_ratio, self.fp16_ratio)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in ratios
        ) or not math.isclose(sum(map(float, ratios)), 1.0, abs_tol=1.0e-6):
            raise ValueError("NVFP4/INT8/FP16 ratios must be nonnegative and sum to one")
        if self.prefix_kv_precision not in ("nvfp4", "int8", "fp16"):
            raise ValueError("prefix_kv_precision must be nvfp4, int8, or fp16")
        if self.prefix_query_precision not in ("int8", "fp16"):
            raise ValueError("prefix_query_precision must be int8 or fp16")


@dataclass(frozen=True)
class NVFP4Calibration:
    """Tensor-level Q/K/V global scales for NVFP4 preparation."""

    q_scale: float = 1.0
    k_scale: float = 1.0
    v_scale: float = 1.0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (self.q_scale, self.k_scale, self.v_scale)
        ):
            raise ValueError("NVFP4 calibration scales must be finite and positive")

    @property
    def scales(self) -> tuple[float, float, float]:
        return tuple(map(float, (self.q_scale, self.k_scale, self.v_scale)))


def anemoi_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    layout: VisualLayout,
    layer: int,
    sparse_config: SparseConfig | None = None,
    quant_config: QuantConfig | None = None,
    calibration: NVFP4Calibration | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Run structured visual inference attention on BSHD Q/K/V tensors."""
    if attn_mask is not None:
        raise ValueError("attn_mask is not supported")
    if dropout_p != 0.0:
        raise ValueError("dropout_p must be 0.0")
    if is_causal:
        raise ValueError("is_causal must be False")
    if scale is not None and not math.isclose(scale, 1.0 / math.sqrt(128)):
        raise ValueError("scale must be None or 1/sqrt(128)")
    if not isinstance(layout, VisualLayout):
        raise TypeError("layout must be VisualLayout")
    if type(layer) is not int or layer < 0:
        raise ValueError("layer must be a nonnegative integer")
    if query.ndim != 4 or query.size(1) != layout.sequence_tokens:
        raise ValueError("Q sequence length must match layout.sequence_tokens")
    sparse = sparse_config or SparseConfig()
    quant = quant_config or QuantConfig()
    resolved_calibration = calibration if calibration is not None else NVFP4Calibration()
    if not isinstance(resolved_calibration, NVFP4Calibration):
        raise TypeError("calibration must be NVFP4Calibration")
    sparsity_ratio = sparse.sparsity_ratio
    for first, last, ratio in sparse.layer_sparsity_bands:
        if first <= layer < last:
            sparsity_ratio = ratio
            break

    from anemoi.layers.attention.mpa.executor import execute_ragged_attention

    return execute_ragged_attention(
        query,
        key,
        value,
        prefix_tokens=layout.prefix_tokens,
        video_shape=layout.video_shape,
        layer=layer,
        query_block_size=sparse.query_block_size,
        sparsity_ratio=sparsity_ratio,
        nvfp4_ratio=quant.nvfp4_ratio,
        int8_ratio=quant.int8_ratio,
        fp16_ratio=quant.fp16_ratio,
        prefix_kv_precision=quant.prefix_kv_precision,
        prefix_query_precision=quant.prefix_query_precision,
        maxpool_weight=float(sparse.maxpool_weight),
        nvfp4_scales=resolved_calibration.scales,
    )


__all__ = [
    "NVFP4Calibration",
    "QuantConfig",
    "SparseConfig",
    "VisualLayout",
    "anemoi_attention",
]
