"""MiniMax-H3 adapter for stripe-compact ragged mixed attention."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from anemoi.layers.attention.mpa._private_h3_layout import _is_exact_h3_packed_qkv

from .native_k64_attention import (
    sm89_ragged_h3_attention,
    sm120_ragged_h3_attention,
)

try:
    from torch.backends.cuda import SDPAParams as _SDPAParams
    from torch.backends.cuda import can_use_cudnn_attention as _can_use_cudnn
except (AttributeError, ImportError):
    _SDPAParams = None
    _can_use_cudnn = None


@dataclass(frozen=True)
class H3MPAConfig:
    """Geometry, schedule, and precision phases for one H3 request."""

    video_shape: tuple[int, int, int] = (37, 24, 42)
    prefix_tokens: int = 951
    sparsity_ratio: float = 0.88
    query_block_size: int = 64
    prefix_kv_precision: str = "auto"
    prefix_query_precision: str = "auto"
    fp8_ratio: float = 0.80
    nvfp4_ratio: float = 0.0
    int8_ratio: float = 0.0
    mxfp8_ratio: float = 0.0
    fp16_ratio: float = 0.20
    dense_first_steps: int = 10
    dense_first_layers: int = 2
    layers_per_step: int = 50
    layer_sparsity_bands: tuple[tuple[int, int, float], ...] = (
        (18, 34, 0.82),
        (34, 50, 0.58),
    )
    layer_precision_bands: tuple[tuple[float | int, ...], ...] = ()
    diag_jensen: bool = False
    enable_anchors: bool = True
    strict: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.video_shape, tuple)
            or len(self.video_shape) != 3
            or any(type(value) is not int or value <= 0 for value in self.video_shape)
        ):
            raise ValueError("video_shape must contain three positive integers")
        if type(self.prefix_tokens) is not int or self.prefix_tokens <= 0:
            raise ValueError("prefix_tokens must be a positive integer")
        if type(self.layers_per_step) is not int or self.layers_per_step <= 0:
            raise ValueError("layers_per_step must be a positive integer")
        if type(self.query_block_size) is not int or self.query_block_size not in (64, 128):
            raise ValueError("query_block_size must be 64 or 128")
        if self.prefix_kv_precision not in (
            "auto",
            "fp16",
            "mxfp8",
            "nvfp4",
            "int8",
        ):
            raise ValueError(
                "prefix_kv_precision must be auto, fp16, mxfp8, nvfp4, or int8"
            )
        if self.prefix_query_precision not in (
            "auto",
            "fp16",
            "int8",
            "mxfp8",
            "nvfp4",
        ):
            raise ValueError(
                "prefix_query_precision must be auto, fp16, int8, mxfp8, or nvfp4"
            )
        if any(
            type(value) is not int or value < 0
            for value in (self.dense_first_steps, self.dense_first_layers)
        ):
            raise ValueError("dense guards must be nonnegative integers")
        if self.dense_first_layers > self.layers_per_step:
            raise ValueError("dense_first_layers exceeds layers_per_step")
        if (
            isinstance(self.sparsity_ratio, bool)
            or not isinstance(self.sparsity_ratio, (int, float))
            or not math.isfinite(float(self.sparsity_ratio))
            or not 0.0 <= float(self.sparsity_ratio) < 1.0
        ):
            raise ValueError("sparsity_ratio must be finite and in [0, 1)")
        sm120_ratios = tuple(
            map(
                float,
                (
                    self.nvfp4_ratio,
                    self.int8_ratio,
                    self.mxfp8_ratio,
                    self.fp16_ratio,
                ),
            )
        )
        legacy_q64 = self.query_block_size == 64 and float(self.fp8_ratio) != 0.0
        if legacy_q64:
            ratios = (float(self.fp8_ratio), float(self.fp16_ratio))
            if (
                any(not math.isfinite(value) or value <= 0.0 for value in ratios)
                or not math.isclose(sum(ratios), 1.0, abs_tol=1.0e-6)
                or self.nvfp4_ratio != 0.0
                or self.int8_ratio != 0.0
                or self.mxfp8_ratio != 0.0
                or self.layer_precision_bands
            ):
                raise ValueError("Q64 requires positive FP8/FP16 ratios summing to one")
        else:
            if (
                float(self.fp8_ratio) != 0.0
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in sm120_ratios
                )
                or not math.isclose(sum(sm120_ratios), 1.0, abs_tol=1.0e-6)
            ):
                raise ValueError(
                    "SM120 requires nonnegative NVFP4/INT8/MXFP8/FP16 "
                    "ratios summing to one"
                )
            if self.int8_ratio > 0.0 and self.mxfp8_ratio > 0.0:
                raise ValueError("INT8 and MXFP8 are alternative middle phases")
        if (
            type(self.diag_jensen) is not bool
            or type(self.enable_anchors) is not bool
            or type(self.strict) is not bool
        ):
            raise TypeError("diag_jensen, enable_anchors, and strict must be bool")

        previous_end = 0
        for band in self.layer_sparsity_bands:
            if not isinstance(band, tuple) or len(band) != 3:
                raise TypeError("layer bands must be (first, last, sparsity)")
            first, last, sparsity = band
            if (
                type(first) is not int
                or type(last) is not int
                or not 0 <= first < last <= self.layers_per_step
                or first < previous_end
            ):
                raise ValueError("layer bands must be sorted, disjoint, and in range")
            if (
                isinstance(sparsity, bool)
                or not isinstance(sparsity, (int, float))
                or not math.isfinite(float(sparsity))
                or not 0.0 <= float(sparsity) < 1.0
            ):
                raise ValueError("layer sparsities must be finite and in [0, 1)")
            previous_end = last

        previous_end = 0
        for band in self.layer_precision_bands:
            if not isinstance(band, tuple) or len(band) not in (5, 6):
                raise TypeError(
                    "layer precision bands must be (first, last, NVFP4, INT8, "
                    "[MXFP8,] FP16)"
                )
            first, last, *precision = band
            if (
                type(first) is not int
                or type(last) is not int
                or not 0 <= first < last <= self.layers_per_step
                or first < previous_end
            ):
                raise ValueError("layer precision bands must be sorted and disjoint")
            values = tuple(map(float, precision))
            if (
                any(not math.isfinite(value) or value < 0.0 for value in values)
                or not math.isclose(sum(values), 1.0, abs_tol=1.0e-6)
            ):
                raise ValueError("layer precision ratios must be nonnegative and sum to one")
            int8 = values[1]
            mxfp8 = values[2] if len(values) == 4 else 0.0
            if int8 > 0.0 and mxfp8 > 0.0:
                raise ValueError("INT8 and MXFP8 are alternative middle phases")
            previous_end = last

    @property
    def video_tokens(self) -> int:
        return math.prod(self.video_shape)

    @property
    def sequence_tokens(self) -> int:
        return self.prefix_tokens + self.video_tokens

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            video_tokens=self.video_tokens,
            sequence_tokens=self.sequence_tokens,
        )
        return result

    def precision(self, layer: int) -> tuple[float, float, float, float]:
        if type(layer) is not int or not 0 <= layer < self.layers_per_step:
            raise ValueError("layer is outside the configured transformer stack")
        result = (
            float(self.nvfp4_ratio),
            float(self.int8_ratio),
            float(self.mxfp8_ratio),
            float(self.fp16_ratio),
        )
        for band in self.layer_precision_bands:
            first, last, *precision = band
            if first <= layer < last:
                if len(precision) == 3:
                    nvfp4, int8, fp16 = precision
                    return float(nvfp4), float(int8), 0.0, float(fp16)
                nvfp4, int8, mxfp8, fp16 = precision
                return tuple(map(float, (nvfp4, int8, mxfp8, fp16)))
        return result


def _dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
    )
    return output.squeeze(0).transpose(0, 1)


def _can_use_direct_packed_views(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: H3MPAConfig,
) -> bool:
    if not _is_exact_h3_packed_qkv(
        q,
        k,
        v,
        video_shape=config.video_shape,
        prefix_tokens=config.prefix_tokens,
    ):
        return False
    if _SDPAParams is None or _can_use_cudnn is None:
        return False
    q_prefix = q[: config.prefix_tokens].unsqueeze(0).transpose(1, 2)
    k_full = k.unsqueeze(0).transpose(1, 2)
    v_full = v.unsqueeze(0).transpose(1, 2)
    return bool(_can_use_cudnn(_SDPAParams(q_prefix, k_full, v_full, None, 0.0, False, False)))


def _target_video_start(video_indices: torch.Tensor, sequence_length: int) -> int:
    if not isinstance(video_indices, torch.Tensor) or video_indices.ndim != 1:
        raise ValueError("video_indices must be rank one")
    if video_indices.numel() == 0:
        return sequence_length
    if video_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("video_indices must have integer dtype")
    breaks = ((video_indices[1:] - video_indices[:-1]) != 1).nonzero()
    start = int(breaks[-1].item()) + 1 if breaks.numel() else 0
    return int(video_indices[start].item())


class H3MPAAttention:
    """Post-Ulysses callable with Sol-compatible dense scheduling."""

    def __init__(self, config: H3MPAConfig):
        if not isinstance(config, H3MPAConfig):
            raise TypeError("config must be H3MPAConfig")
        self.config = config
        self.step = -1
        self.layer = 0
        self.request = -1
        self._previous_timestep: float | None = None
        self._direction = 0
        self._observed_sequence: int | None = None
        self.dense_calls = 0
        self.mpa_calls = 0
        self.last_request_mpa_calls: int | None = None
        self._mpa_at_request_start = 0
        self._last_closed_request = -1

    @staticmethod
    def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
            raise TypeError("Q/K/V must be tensors")
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
            raise ValueError("Q/K/V must share [sequence,heads,128]")
        if (
            q.device.type != "cuda"
            or k.device != q.device
            or v.device != q.device
            or q.dtype not in (torch.float16, torch.bfloat16)
            or k.dtype != q.dtype
            or v.dtype != q.dtype
            or q.shape[-1] != 128
            or any(tensor.stride(-1) != 1 for tensor in (q, k, v))
        ):
            raise ValueError("Q/K/V must be same-dtype CUDA tensors with head_dim=128")

    @staticmethod
    def dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        H3MPAAttention._validate_qkv(q, k, v)
        return _dense_attention(q, k, v)

    def observe(
        self,
        *,
        timestep: torch.Tensor | None,
        video_indices: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> None:
        sequence = int(position_ids.shape[0])
        self._observed_sequence = sequence
        if sequence != self.config.sequence_tokens:
            raise ValueError(
                f"observed {sequence} tokens, expected {self.config.sequence_tokens}"
            )
        if video_indices is not None:
            prefix = _target_video_start(video_indices, sequence)
            if prefix != self.config.prefix_tokens:
                raise ValueError(
                    f"observed prefix {prefix}, expected {self.config.prefix_tokens}"
                )

        current = None
        if timestep is not None:
            current = float(timestep.detach().float().max().item())
        previous, self._previous_timestep = self._previous_timestep, current
        reversed_request = False
        if previous is not None and current is not None:
            delta = current - previous
            if abs(delta) > 1.0e-6:
                sign = 1 if delta > 0 else -1
                if self._direction == 0:
                    self._direction = sign
                elif sign != self._direction:
                    reversed_request = True
        if self.step < 0 or reversed_request:
            self.close_request()
            self.request += 1
            self.step = 0
            self._direction = 0
        else:
            self.step += 1
        self.layer = 0

    def close_request(self) -> None:
        if self.request < 0 or self.request == self._last_closed_request:
            return
        self.last_request_mpa_calls = self.mpa_calls - self._mpa_at_request_start
        self._mpa_at_request_start = self.mpa_calls
        self._last_closed_request = self.request
        if (
            self.step + 1 <= self.config.dense_first_steps
            or self.last_request_mpa_calls > 0
        ):
            return
        message = "request passed the dense-first schedule without an MPA call"
        if self.config.strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _sparsity(self, layer: int) -> float:
        if type(layer) is not int or not 0 <= layer < self.config.layers_per_step:
            raise ValueError("layer is outside the configured transformer stack")
        value = float(self.config.sparsity_ratio)
        for first, last, sparsity in self.config.layer_sparsity_bands:
            if first <= layer < last:
                return float(sparsity)
        return value

    def mpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        layer: int,
    ) -> torch.Tensor:
        self._validate_qkv(q, k, v)
        if q.shape[0] != self.config.sequence_tokens:
            raise ValueError("packed sequence length does not match the H3 request")
        if _can_use_direct_packed_views(q, k, v, self.config):
            q_full, k_full, v_full = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        else:
            q_full = q.unsqueeze(0).contiguous()
            k_full = k.unsqueeze(0).contiguous()
            v_full = v.unsqueeze(0).contiguous()
        if self.config.query_block_size == 64 and self.config.fp8_ratio > 0.0:
            output = sm89_ragged_h3_attention(
                q_full,
                k_full,
                v_full,
                prefix_tokens=self.config.prefix_tokens,
                sparsity_ratio=self._sparsity(layer),
                retained_fp8_ratio=self.config.fp8_ratio,
                retained_fp16_ratio=self.config.fp16_ratio,
                video_shape=self.config.video_shape,
                diag_jensen=self.config.diag_jensen,
                enable_anchors=self.config.enable_anchors,
            )
        else:
            nvfp4, int8, mxfp8, fp16 = self.config.precision(layer)
            output = sm120_ragged_h3_attention(
                q_full,
                k_full,
                v_full,
                prefix_tokens=self.config.prefix_tokens,
                sparsity_ratio=self._sparsity(layer),
                retained_nvfp4_ratio=nvfp4,
                retained_int8_ratio=int8,
                retained_mxfp8_ratio=mxfp8,
                retained_fp16_ratio=fp16,
                video_shape=self.config.video_shape,
                layer=layer,
                query_block_size=self.config.query_block_size,
                prefix_kv_precision=self.config.prefix_kv_precision,
                prefix_query_precision=self.config.prefix_query_precision,
                diag_jensen=self.config.diag_jensen,
                enable_anchors=self.config.enable_anchors,
            )
        return output.squeeze(0)

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.config.strict and self.request < 0:
            raise RuntimeError("the H3 layout observer did not run")
        layer = self.layer
        self.layer += 1
        if self.step < self.config.dense_first_steps or layer < self.config.dense_first_layers:
            self.dense_calls += 1
            return self.dense(q, k, v)
        self.mpa_calls += 1
        return self.mpa(q, k, v, layer=layer)

    def stats(self) -> dict[str, Any]:
        total = self.dense_calls + self.mpa_calls
        return {
            "config": self.config.to_dict(),
            "routing_block_size": self.config.query_block_size,
            "request": self.request,
            "step": self.step,
            "layer": self.layer,
            "observed_sequence": self._observed_sequence,
            "dense_calls": self.dense_calls,
            "mpa_calls": self.mpa_calls,
            "mpa_calls_measured_request": self.last_request_mpa_calls,
            "mpa_fraction": self.mpa_calls / total if total else None,
        }


def install_layout_observer(
    transformer: torch.nn.Module,
    attention: H3MPAAttention,
):
    """Observe request/step state before the post-Ulysses attention callable."""

    if not isinstance(attention, H3MPAAttention):
        raise TypeError("attention must be H3MPAAttention")

    def pre_hook(_module, _args, kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            attention.observe(
                timestep=kwargs.get("timestep"),
                video_indices=kwargs.get("video_indices"),
                position_ids=position_ids,
            )
        return None

    return transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)


__all__ = ["H3MPAAttention", "H3MPAConfig", "install_layout_observer"]
