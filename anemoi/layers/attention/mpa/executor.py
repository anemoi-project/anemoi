"""Generic native Q64/Q128 executors for structured visual attention."""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch
import torch.nn.functional as F

from anemoi.layers.attention.mpa.backends.sm89_k64 import (
    assemble_h3_k64_output,
    native_k64_mixed_attention,
    pack_h3_k64_qkv,
    prepare_h3_sm89_int8_operands,
    prepare_k64_fp8_operands,
    prepare_prefix_q_int8,
    sm89_h3_draft_probability,
    sm89_h3_materialize_route,
    sm89_h3_route_precision,
    sm89_q64_prefix_int8_attention,
)
from anemoi.layers.attention.mpa.backends.sm120_q64 import (
    prepare_h3_sm120_operands,
    prepare_mxfp8,
    prepare_q64_nvfp4,
    sm120_h3_draft_probability,
    sm120_h3_k_tail_r1_probability,
    sm120_h3_k_tail_r2_probability,
    sm120_h3_materialize_route,
    sm120_h3_route_precision,
    sm120_q64_fp16_attention,
    sm120_q64_int8_fp16_attention,
    sm120_q64_mxfp8_attention,
    sm120_q64_nv_int8_fp16_attention,
    sm120_q64_nv_mxfp8_fp16_attention,
    sm120_q64_nvfp4_fp16_attention,
    sm120_q64_prefix_int8_attention,
)
from anemoi.layers.attention.mpa.backends.sm120_q128 import (
    prepare_q128_nvfp4,
    sm120_q128_fp16_attention,
    sm120_q128_int8_fp16_attention,
    sm120_q128_mxfp8_attention,
    sm120_q128_nv_int8_fp16_attention,
    sm120_q128_nv_mxfp8_fp16_attention,
    sm120_q128_nvfp4_fp16_attention,
    sm120_q128_prefix_int8_attention,
)
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.routing import (
    RaggedRoutePlan,
    _route_counts,
    draft_probability,
    route_probability,  # noqa: F401 - kept for the historical compatibility alias
)

_BLOCK = 64
_SM120_PHASE_NAMES = ("nvfp4", "int8", "mxfp8", "fp16")
_SM120_PREFIX_QUERY_PRECISIONS = ("auto", "fp16", "int8", "mxfp8", "nvfp4")


def _warmup_sync(stage: str, device: torch.device) -> None:
    if os.environ.get("ANEMOI_MPA_WARMUP_SYNC") != "1":
        return
    try:
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(f"native attention warm-up failed after {stage}") from exc


@lru_cache(maxsize=8)
def _sm120_prefix_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device)


@lru_cache(maxsize=8)
def _sm89_prefix_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device)


def _pool_query(
    query_fp16: torch.Tensor,
    counts: torch.Tensor,
    block: int = _BLOCK,
    maximum: bool = False,
) -> torch.Tensor:
    batch, heads, tokens, head_dim = query_fp16.shape
    blocks = tokens // block
    blocked = query_fp16.view(batch, heads, blocks, block, head_dim)
    if maximum:
        valid = torch.arange(block, device=query_fp16.device).view(
            1, 1, 1, block, 1
        ) < counts.view(batch, 1, blocks, 1, 1)
        return (
            blocked.masked_fill(~valid, -torch.inf)
            .amax(dim=3)
            .to(torch.float16)
            .contiguous()
        )
    denominator = counts.to(torch.float32).view(batch, 1, blocks, 1)
    return (
        blocked.sum(dim=3, dtype=torch.float32)
        .div_(denominator)
        .to(torch.float16)
        .contiguous()
    )


def _compose_q128_phase_route(
    logical_ids: torch.Tensor,
    nvfp4_counts: torch.Tensor,
    int8_counts: torch.Tensor,
    fp16_counts: torch.Tensor,
    *,
    prefix_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand logical K128 IDs while leaving prefix synthesis to the kernel."""

    if logical_ids.dtype != torch.int32 or logical_ids.ndim != 4:
        raise TypeError("logical_ids must be rank-four int32")
    if any(
        counts.shape != logical_ids.shape[:-1] or counts.dtype != torch.int32
        for counts in (nvfp4_counts, int8_counts, fp16_counts)
    ):
        raise ValueError("Q128 phase counts must match logical route rows")
    if type(prefix_blocks) is not int or prefix_blocks < 0:
        raise ValueError("prefix_blocks must be nonnegative")
    first = prefix_blocks + logical_ids.mul(2)
    video = torch.stack((first, first + 1), dim=-1).flatten(-2)
    prefix = torch.arange(prefix_blocks, device=logical_ids.device, dtype=torch.int32).view(
        1, 1, 1, -1
    )
    prefix = prefix.expand(*logical_ids.shape[:-1], -1)
    return (
        torch.cat((video, prefix), dim=-1).contiguous(),
        nvfp4_counts.mul(2).contiguous(),
        int8_counts.mul(2).contiguous(),
        fp16_counts.mul(2).add(prefix_blocks).contiguous(),
    )


def _sm120_phase_config(
    ratios: tuple[float, float, float, float],
) -> tuple[str, ...]:
    values = tuple(map(float, ratios))
    if (
        len(values) != 4
        or any(not math.isfinite(value) or value < 0.0 for value in values)
        or not math.isclose(sum(values), 1.0, abs_tol=1.0e-6)
    ):
        raise ValueError("SM120 phase ratios must be nonnegative and sum to one")
    return tuple(name for name, value in zip(_SM120_PHASE_NAMES, values) if value > 0.0)


def _resolve_sm120_prefix_phase(video_phases: tuple[str, ...], prefix_kv_precision: str) -> str:
    if prefix_kv_precision == "auto":
        return "fp16" if "fp16" in video_phases else video_phases[0]
    if prefix_kv_precision not in _SM120_PHASE_NAMES:
        raise ValueError("prefix_kv_precision must be auto, fp16, mxfp8, nvfp4, or int8")
    return prefix_kv_precision


def _resolve_sm120_prefix_query_precision(
    video_phases: tuple[str, ...], prefix_query_precision: str
) -> str:
    if prefix_query_precision not in _SM120_PREFIX_QUERY_PRECISIONS:
        raise ValueError("prefix_query_precision must be auto, fp16, int8, mxfp8, or nvfp4")
    if prefix_query_precision == "auto":
        return "int8" if "int8" in video_phases else "fp16"
    return prefix_query_precision


def _resolve_sm89_prefix_precision(value: str, *, field: str) -> str:
    """Keep legacy auto/fp16 while exposing the shared explicit INT8 cell."""

    if value == "auto":
        return "fp16"
    if value not in ("fp16", "int8"):
        raise ValueError(f"SM89 {field} must be auto, fp16, or int8")
    return value


def _route_sm89_probability(
    probability: torch.Tensor,
    anchors: torch.Tensor | None,
    anchor_ids: torch.Tensor | None,
    *,
    anchor_count: int,
    sparsity_ratio: float,
    int8_ratio: float,
    fp16_ratio: float,
) -> RaggedRoutePlan:
    n16, n8, n4 = _route_counts(
        probability.size(-2) * probability.size(-1),
        sparsity_ratio=sparsity_ratio,
        nvfp4_ratio=0.0,
        fp8_ratio=int8_ratio,
        fp16_ratio=fp16_ratio,
    )
    if n4 != 0:
        raise RuntimeError("SM89 route unexpectedly received an NVFP4 budget")
    if (anchors is None) != (anchor_ids is None):
        raise ValueError("anchors and anchor_ids must be enabled together")
    if anchors is not None and anchor_count > n8:
        raise ValueError("anchor_count exceeds the configured lowest-precision budget")
    block_ids, low_counts, fp8_counts, fp16_counts = sm89_h3_route_precision(
        probability,
        n16,
        n8,
        anchors,
        anchor_ids,
        anchor_count,
    )
    return RaggedRoutePlan(
        block_ids=block_ids,
        nvfp4_counts=low_counts,
        fp8_counts=fp8_counts,
        fp16_counts=fp16_counts,
        fp16_blocks_per_head=n16,
        fp8_blocks_per_head=n8,
        nvfp4_blocks_per_head=0,
    )


def _active_sm120_phases(video_phases: tuple[str, ...], prefix_phase: str) -> tuple[str, ...]:
    return tuple(
        phase for phase in _SM120_PHASE_NAMES if phase in video_phases or phase == prefix_phase
    )


def _compose_sm120_phase_route(
    logical_ids: torch.Tensor,
    nvfp4_counts: torch.Tensor,
    middle_counts: torch.Tensor,
    fp16_counts: torch.Tensor,
    *,
    query_block_size: int,
    prefix_blocks: int,
    prefix_phase: str,
    active_phases: tuple[str, ...],
    legacy_auto_prefix_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    factor = query_block_size // _BLOCK
    first = prefix_blocks + logical_ids.mul(factor)
    video = first if factor == 1 else torch.stack((first, first + 1), dim=-1).flatten(-2)
    nv = nvfp4_counts.mul(factor).contiguous()
    middle = middle_counts.mul(factor).contiguous()
    high = fp16_counts.mul(factor).contiguous()
    insert_after = {
        "nvfp4": torch.zeros_like(nv),
        "int8": nv,
        "mxfp8": nv,
        "fp16": nv + middle + high,
    }.get(prefix_phase)
    if insert_after is None or prefix_phase not in active_phases:
        raise ValueError("prefix phase must be active")
    insert_after = insert_after.to(torch.int64).unsqueeze(-1)
    prefix = (
        torch.arange(prefix_blocks, device=video.device, dtype=torch.int32)
        .view(*([1] * (video.ndim - 1)), -1)
        .expand(*video.shape[:-1], -1)
    )
    if legacy_auto_prefix_order and prefix_phase == "fp16":
        block_ids = torch.cat((video, prefix), dim=-1)
    elif prefix_phase == active_phases[0]:
        block_ids = torch.cat((prefix, video), dim=-1)
    else:
        positions = torch.arange(video.size(-1), device=video.device, dtype=torch.int64).view(
            *([1] * (video.ndim - 1)), -1
        )
        video_destinations = positions + (positions >= insert_after) * prefix_blocks
        block_ids = torch.empty(
            (*video.shape[:-1], video.size(-1) + prefix_blocks),
            device=video.device,
            dtype=torch.int32,
        )
        block_ids.scatter_(-1, video_destinations.expand_as(video), video)
        prefix_destinations = insert_after + torch.arange(
            prefix_blocks, device=video.device, dtype=torch.int64
        )
        block_ids.scatter_(-1, prefix_destinations, prefix)
    if prefix_phase == "nvfp4":
        nv = nv.add(prefix_blocks).contiguous()
    elif prefix_phase in ("int8", "mxfp8"):
        middle = middle.add(prefix_blocks).contiguous()
    else:
        high = high.add(prefix_blocks).contiguous()
    return (
        block_ids.contiguous(),
        nv,
        middle,
        high
        if "fp16" in active_phases
        else torch.empty(0, dtype=torch.int32, device=logical_ids.device),
    )


def _route_sm120_probability(
    probability: torch.Tensor,
    anchors: torch.Tensor | None,
    anchor_ids: torch.Tensor | None,
    *,
    anchor_count: int,
    sparsity_ratio: float,
    ratios: tuple[float, float, float, float],
) -> RaggedRoutePlan:
    n16, n8, n4 = _route_counts(
        probability.size(-2) * probability.size(-1),
        sparsity_ratio=sparsity_ratio,
        nvfp4_ratio=ratios[0],
        fp8_ratio=ratios[1] + ratios[2],
        fp16_ratio=ratios[3],
    )
    if anchors is not None and anchor_count > (n4 or n8 or n16):
        raise ValueError("anchor_count exceeds the configured lowest-precision budget")
    if (anchors is None) != (anchor_ids is None):
        raise ValueError("anchors and anchor_ids must be enabled together")
    block_ids, nv_counts, middle_counts, high_counts = sm120_h3_route_precision(
        probability, n16, n8, n4, anchors, anchor_ids, anchor_count
    )
    return RaggedRoutePlan(
        block_ids=block_ids,
        nvfp4_counts=nv_counts,
        fp8_counts=middle_counts,
        fp16_counts=high_counts,
        fp16_blocks_per_head=n16,
        fp8_blocks_per_head=n8,
        nvfp4_blocks_per_head=n4,
    )


def _expand_q128_valid_k(
    prefix_counts: torch.Tensor,
    video_counts: torch.Tensor,
) -> torch.Tensor:
    halves = torch.stack(
        (video_counts.clamp_max(_BLOCK), (video_counts - _BLOCK).clamp_min(0)),
        dim=-1,
    )
    return torch.cat((prefix_counts, halves.flatten(1)), dim=1).contiguous()


def _compose_q128_fp16_route(
    logical_ids: torch.Tensor,
    fp16_counts: torch.Tensor,
    *,
    prefix_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = prefix_blocks + logical_ids.mul(2)
    video = torch.stack((first, first + 1), dim=-1).flatten(-2)
    prefix = (
        torch.arange(prefix_blocks, device=logical_ids.device, dtype=torch.int32)
        .view(1, 1, 1, -1)
        .expand(*logical_ids.shape[:-1], -1)
    )
    return (
        torch.cat((prefix, video), dim=-1).contiguous(),
        fp16_counts.mul(2).add(prefix_blocks).contiguous(),
    )


@lru_cache(maxsize=128)
def _nvfp4_global_scales(
    device: torch.device, scales: tuple[float, float, float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.tensor(scale, device=device, dtype=torch.float32) for scale in scales)


def _prepare_h3_sm120_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    video_token_indices: torch.Tensor,
    video_slot_valid: torch.Tensor,
    video_valid_counts: torch.Tensor,
    prefix_tokens: int,
    query_block_size: int,
    phases: tuple[str, ...],
    has_prefix_query_int8: bool = False,
    has_maxpool: bool = False,
    global_scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[object, ...]:
    prepared = prepare_h3_sm120_operands(
        query,
        key,
        value,
        video_token_indices,
        video_slot_valid,
        video_valid_counts,
        prefix_tokens=prefix_tokens,
        query_block_size=query_block_size,
        has_nvfp4="nvfp4" in phases,
        has_int8="int8" in phases,
        has_mxfp8="mxfp8" in phases,
        has_fp16="fp16" in phases,
        has_prefix_query_int8=has_prefix_query_int8,
        has_maxpool=has_maxpool,
        global_scales=global_scales,
    )
    return (
        prepared[0],
        prepared[1],
        *prepared[2:5],
        prepared[5:11],
        prepared[11:17],
        prepared[17:23],
        prepared[23:25],
        prepared[25],
        prepared[26],
    )


def _prepare_int8_operands(
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    query_block_size: int,
) -> tuple[torch.Tensor, ...]:
    q8, k8, v8, q8_scale, k8_scale, v8_scale = prepare_k64_fp8_operands(
        query_fp16, key_fp16, value_fp16, query_block=query_block_size
    )
    return q8, q8_scale, k8, k8_scale, v8, v8_scale


def _run_sm120_phases(
    *,
    query_block_size: int,
    ratios: tuple[float, float, float, float],
    prefix_phase: str | None = None,
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    block_ids: torch.Tensor,
    nvfp4_counts: torch.Tensor,
    middle_counts: torch.Tensor,
    fp16_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
    layer: int,
    fp16_prefix_blocks: int,
    prepared_nv_operands: tuple[torch.Tensor, ...] | None = None,
    prepared_int8_operands: tuple[torch.Tensor, ...] | None = None,
    prepared_mxfp8_operands: tuple[torch.Tensor, ...] | None = None,
    prepared_global_scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    nvfp4_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    video_phases = _sm120_phase_config(ratios)
    phases = (
        video_phases if prefix_phase is None else _active_sm120_phases(video_phases, prefix_phase)
    )
    active_fp16 = "fp16" in phases
    global_scales = prepared_global_scales
    if global_scales is None and "nvfp4" in phases:
        global_scales = _nvfp4_global_scales(query_fp16.device, nvfp4_scales)
    nv_operands = prepared_nv_operands
    if global_scales is not None and nv_operands is None:
        prepare_nvfp4 = prepare_q64_nvfp4 if query_block_size == 64 else prepare_q128_nvfp4
        nv_operands = prepare_nvfp4(query_fp16, key_fp16, value_fp16, *global_scales)
    int8_operands = prepared_int8_operands
    if "int8" in phases and int8_operands is None:
        int8_operands = _prepare_int8_operands(query_fp16, key_fp16, value_fp16, query_block_size)
    mxfp8_operands = prepared_mxfp8_operands
    if "mxfp8" in phases and mxfp8_operands is None:
        mxfp8_operands = prepare_mxfp8(query_fp16, key_fp16, value_fp16)
    _warmup_sync("SM120 precision preparation", query_fp16.device)

    prefix = fp16_prefix_blocks if active_fp16 and prefix_phase in (None, "fp16") else 0
    if phases == ("fp16",):
        function = sm120_q64_fp16_attention if query_block_size == 64 else sm120_q128_fp16_attention
        return function(
            query_fp16,
            key_fp16,
            value_fp16,
            block_ids,
            fp16_counts,
            valid_k_counts,
        )
    if phases in (("int8",), ("int8", "fp16")):
        assert int8_operands is not None
        function = (
            sm120_q64_int8_fp16_attention
            if query_block_size == 64
            else sm120_q128_int8_fp16_attention
        )
        return function(
            int8_operands,
            query_fp16,
            key_fp16,
            value_fp16,
            block_ids,
            middle_counts,
            fp16_counts,
            valid_k_counts,
            fp16_prefix_blocks=prefix,
            active_fp16=active_fp16,
        )
    if phases in (("mxfp8",), ("mxfp8", "fp16")):
        assert mxfp8_operands is not None
        function = (
            sm120_q64_mxfp8_attention if query_block_size == 64 else sm120_q128_mxfp8_attention
        )
        if query_block_size == 64:
            return function(
                *mxfp8_operands,
                query_fp16,
                key_fp16,
                value_fp16,
                block_ids,
                middle_counts,
                fp16_counts,
                valid_k_counts,
                fp16_prefix_blocks=prefix,
                active_fp16=active_fp16,
            )
        return function(
            mxfp8_operands,
            query_fp16,
            key_fp16,
            value_fp16,
            block_ids,
            middle_counts,
            fp16_counts,
            valid_k_counts,
            fp16_prefix_blocks=prefix,
            active_fp16=active_fp16,
        )
    if phases in (("nvfp4",), ("nvfp4", "fp16")):
        assert nv_operands is not None and global_scales is not None
        function = (
            sm120_q64_nvfp4_fp16_attention
            if query_block_size == 64
            else sm120_q128_nvfp4_fp16_attention
        )
        return function(
            nv_operands,
            query_fp16,
            key_fp16,
            value_fp16,
            block_ids,
            nvfp4_counts,
            fp16_counts,
            valid_k_counts,
            global_scales,
            fp16_prefix_blocks=prefix,
            active_fp16=active_fp16,
        )
    if phases in (("nvfp4", "int8"), ("nvfp4", "int8", "fp16")):
        assert nv_operands is not None and int8_operands is not None
        assert global_scales is not None
        function = (
            sm120_q64_nv_int8_fp16_attention
            if query_block_size == 64
            else sm120_q128_nv_int8_fp16_attention
        )
        return function(
            nv_operands,
            int8_operands,
            query_fp16,
            key_fp16,
            value_fp16,
            block_ids,
            nvfp4_counts,
            middle_counts,
            fp16_counts,
            valid_k_counts,
            global_scales,
            fp16_prefix_blocks=prefix,
            active_fp16=active_fp16,
        )
    if phases in (("nvfp4", "mxfp8"), ("nvfp4", "mxfp8", "fp16")):
        assert nv_operands is not None and mxfp8_operands is not None
        assert global_scales is not None
        function = (
            sm120_q64_nv_mxfp8_fp16_attention
            if query_block_size == 64
            else sm120_q128_nv_mxfp8_fp16_attention
        )
        return function(
            nv_operands,
            mxfp8_operands,
            query_fp16,
            key_fp16,
            value_fp16,
            block_ids,
            nvfp4_counts,
            middle_counts,
            fp16_counts,
            valid_k_counts,
            global_scales,
            fp16_prefix_blocks=prefix,
            active_fp16=active_fp16,
        )
    raise RuntimeError(f"SM120 Q{query_block_size} {' -> '.join(phases)} has no native operator")


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
    retained_int8_ratio: float,
    retained_fp16_ratio: float,
    video_shape: tuple[int, int, int],
    query_block_size: int = 64,
    prefix_kv_precision: str = "auto",
    prefix_query_precision: str = "auto",
    smooth_k: bool = False,
    diag_jensen: bool = False,
    maxpool_weight: float = 0.0,
    enable_anchors: bool = False,
) -> torch.Tensor:
    """Run ragged 2-D routing through the native Q64/Q128 x K64 kernel."""

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
    if query_block_size not in (64, 128):
        raise ValueError("query_block_size must be 64 or 128")
    if type(prefix_tokens) is not int or not 0 <= prefix_tokens < q_bshd.size(1):
        raise ValueError("prefix_tokens must be nonnegative and precede the video tokens")
    if (
        not isinstance(video_shape, tuple)
        or len(video_shape) != 3
        or any(type(value) is not int or value <= 0 for value in video_shape)
        or math.prod(video_shape) != q_bshd.size(1) - prefix_tokens
    ):
        raise ValueError("video_shape must match the post-prefix token count")
    if any(type(value) is not bool for value in (diag_jensen, enable_anchors, smooth_k)):
        raise TypeError("diag_jensen, enable_anchors, and smooth_k must be bool")
    if isinstance(maxpool_weight, bool) or not isinstance(
        maxpool_weight, (int, float)
    ):
        raise TypeError("maxpool_weight must be a real number")
    if not math.isfinite(float(maxpool_weight)) or not 0.0 <= float(
        maxpool_weight
    ) <= 1.0:
        raise ValueError("maxpool_weight must be finite and in [0, 1]")
    if maxpool_weight != 0.0 and diag_jensen:
        raise ValueError("maxpool_weight cannot be combined with diag_jensen")
    ratios = (float(retained_int8_ratio), float(retained_fp16_ratio))
    if (
        any(not math.isfinite(value) or value < 0.0 for value in ratios)
        or ratios[0] <= 0.0
        or not math.isclose(sum(ratios), 1.0, abs_tol=1.0e-6)
    ):
        raise ValueError(
            "INT8/FP16 ratios require positive INT8, nonnegative FP16, and must sum to one"
        )

    resolved_prefix_kv = _resolve_sm89_prefix_precision(
        prefix_kv_precision, field="prefix_kv_precision"
    )
    resolved_prefix_query = _resolve_sm89_prefix_precision(
        prefix_query_precision, field="prefix_query_precision"
    )
    prefix_kv_int8 = prefix_tokens == 0 or resolved_prefix_kv == "int8"
    prefix_query_int8 = prefix_tokens > 0 and resolved_prefix_query == "int8"

    frames, height, width = video_shape
    current_stream = torch.cuda.current_stream(q_bshd.device)
    prefix_stream = _sm89_prefix_stream(q_bshd.device)
    prefix_output = q_bshd[:, :0].permute(0, 2, 1, 3)
    if prefix_tokens:
        prefix_stream.wait_stream(current_stream)
        with torch.cuda.stream(prefix_stream):
            if prefix_query_int8:
                # This reads the original strided BHSD view directly and overlaps
                # with video packing plus the shared K/V quantization below.
                prefix_q8, prefix_q_scale = prepare_prefix_q_int8(
                    q_bshd.permute(0, 2, 1, 3), prefix_tokens
                )
            else:
                prefix_output = _dense_prefix_sdpa(q_bshd[:, :prefix_tokens], k_bshd, v_bshd)
    layout = materialize_ragged_2d_layout(
        q_bshd.device,
        frames=frames,
        height=height,
        width=width,
        logical_block=query_block_size,
        enable_anchors=enable_anchors,
    )
    (
        q_pool,
        k_pool,
        q_packed,
        key_fp16,
        value_fp16,
        q8,
        k8,
        v8,
        q8_scale,
        k8_scale,
        v8_scale,
        k_mean,
        q_max_pool,
        k_max_pool,
    ) = prepare_h3_sm89_int8_operands(
        q_bshd.permute(0, 2, 1, 3),
        k_bshd.permute(0, 2, 1, 3),
        v_bshd.permute(0, 2, 1, 3),
        layout.indices,
        layout.slot_valid,
        layout.counts,
        prefix_tokens=prefix_tokens,
        query_block_size=query_block_size,
        smooth_k=smooth_k,
        has_maxpool=maxpool_weight != 0.0,
    )
    _warmup_sync("single-load QKV/INT8 preparation", q_bshd.device)
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
    valid_k = (
        _expand_q128_valid_k(prefix_counts, video_counts)
        if query_block_size == 128
        else torch.cat((prefix_counts, video_counts), dim=1).contiguous()
    )
    key_video = key_fp16[:, :, prefix_capacity:]

    prepared_operands = (q8, k8, v8, q8_scale, k8_scale, v8_scale)
    if smooth_k:
        prepared_operands += (k_mean,)
    del q8, k8, v8, q8_scale, k8_scale, v8_scale, k_mean
    if prefix_query_int8:
        prefix_stream.wait_stream(current_stream)
        with torch.cuda.stream(prefix_stream):
            prefix_output = sm89_q64_prefix_int8_attention(
                prefix_q8,
                prefix_q_scale,
                prepared_operands,
                valid_k,
                prefix_tokens,
            )

    moments = diag_jensen
    q_second = None
    k_second = None
    if moments:
        denominator = video_counts.to(torch.float32).view(1, 1, video_blocks, 1)
        q_second = (
            q_packed.view(1, heads, video_blocks, query_block_size, -1)
            .float()
            .square_()
            .sum(dim=3)
            .div_(denominator)
            .to(torch.float16)
            .contiguous()
        )
        k_second = (
            key_video.view(1, heads, video_blocks, query_block_size, -1)
            .float()
            .square_()
            .sum(dim=3)
            .div_(denominator)
            .to(torch.float16)
            .contiguous()
        )
    if diag_jensen:
        probability = draft_probability(
            q_pool,
            k_pool,
            q_second=q_second,
            k_second=k_second,
        )
    elif maxpool_weight == 0.0:
        # Preserve the exact legacy Mean-only route when the feature is off.
        probability = draft_probability(q_pool, k_pool)
    else:
        probability = sm89_h3_draft_probability(
            q_pool,
            k_pool,
            q_max_pool,
            k_max_pool,
            maxpool_weight=maxpool_weight,
        )
    route = _route_sm89_probability(
        probability,
        layout.anchors,
        layout.anchor_ids,
        anchor_count=layout.anchor_count,
        sparsity_ratio=float(sparsity_ratio),
        int8_ratio=ratios[0],
        fp16_ratio=ratios[1],
    )
    ids, fp8_counts, fp16_counts = sm89_h3_materialize_route(
        route.block_ids,
        route.nvfp4_counts,
        route.fp8_counts,
        route.fp16_counts,
        query_block=query_block_size,
        prefix_blocks=prefix_blocks,
        prefix_int8=prefix_kv_int8,
    )
    _warmup_sync("native DraftMap route and materialization", q_bshd.device)

    del prefix_counts, video_counts
    del key_video, q_pool, k_pool, q_max_pool, k_max_pool
    del q_second, k_second, probability, route
    video_output, _ = native_k64_mixed_attention(
        q_packed,
        key_fp16,
        value_fp16,
        ids,
        fp8_counts,
        ids,
        fp16_counts,
        valid_k,
        fp16_prefix_blocks=0 if prefix_kv_int8 else prefix_blocks,
        prepared_operands=prepared_operands,
        active_int8=ratios[0] > 0.0 or prefix_kv_int8,
        active_fp16=ratios[1] > 0.0 or not prefix_kv_int8,
        query_block=query_block_size,
    )
    _warmup_sync("mixed attention", q_bshd.device)
    del q_packed, key_fp16, value_fp16, valid_k, ids
    current_stream.wait_stream(prefix_stream)
    prefix_output.record_stream(current_stream)
    output = assemble_h3_k64_output(
        prefix_output,
        video_output,
        layout.inverse,
        output_dtype=q_bshd.dtype,
    )
    _warmup_sync("output assembly", q_bshd.device)
    return output


def sm120_ragged_h3_attention(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    *,
    prefix_tokens: int,
    sparsity_ratio: float,
    retained_nvfp4_ratio: float,
    retained_int8_ratio: float,
    retained_fp16_ratio: float,
    retained_mxfp8_ratio: float = 0.0,
    video_shape: tuple[int, int, int],
    layer: int,
    query_block_size: int = 128,
    prefix_kv_precision: str = "auto",
    prefix_query_precision: str = "auto",
    draftmap_proxy: str = "mean",
    diag_jensen: bool = False,
    maxpool_weight: float = 0.0,
    enable_anchors: bool = False,
    profile_nvtx: bool = False,
    nvfp4_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Run per-frame ragged Q64/Q128 routing through native SM120 phases."""

    if profile_nvtx:
        torch.cuda.nvtx.range_push("anemoi.complete")
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
    if torch.cuda.get_device_capability(q_bshd.device) != (12, 0):
        raise RuntimeError("the microscaling path requires SM120")
    if query_block_size not in (64, 128):
        raise ValueError("query_block_size must be 64 or 128")
    if draftmap_proxy not in ("mean", "k_tail_r1", "k_tail_r2"):
        raise ValueError("draftmap_proxy must be mean, k_tail_r1, or k_tail_r2")
    if isinstance(maxpool_weight, bool) or not isinstance(
        maxpool_weight, (int, float)
    ):
        raise TypeError("maxpool_weight must be a real number")
    if not math.isfinite(float(maxpool_weight)) or not 0.0 <= float(
        maxpool_weight
    ) <= 1.0:
        raise ValueError("maxpool_weight must be finite and in [0, 1]")
    if maxpool_weight != 0.0 and (
        diag_jensen or draftmap_proxy in ("k_tail_r1", "k_tail_r2")
    ):
        raise ValueError(
            "maxpool_weight cannot be combined with diag_jensen or K-tail"
        )
    if draftmap_proxy in ("k_tail_r1", "k_tail_r2") and query_block_size != 64:
        raise ValueError("K-tail requires the Q64 SM120 path")
    if draftmap_proxy in ("k_tail_r1", "k_tail_r2") and diag_jensen:
        raise ValueError("K-tail cannot be combined with diag_jensen")
    if type(prefix_tokens) is not int or not 0 <= prefix_tokens < q_bshd.size(1):
        raise ValueError("prefix_tokens must be nonnegative and precede the video tokens")
    if (
        not isinstance(video_shape, tuple)
        or len(video_shape) != 3
        or any(type(value) is not int or value <= 0 for value in video_shape)
        or math.prod(video_shape) != q_bshd.size(1) - prefix_tokens
    ):
        raise ValueError("video_shape must match the post-prefix token count")
    if any(type(value) is not bool for value in (diag_jensen, enable_anchors, profile_nvtx)):
        raise TypeError("attention switches must be bool")
    ratios = tuple(
        map(
            float,
            (
                retained_nvfp4_ratio,
                retained_int8_ratio,
                retained_mxfp8_ratio,
                retained_fp16_ratio,
            ),
        )
    )
    video_phases = _sm120_phase_config(ratios)
    resolved_prefix_query_precision = _resolve_sm120_prefix_query_precision(
        video_phases, prefix_query_precision
    )
    native_prefix_int8 = prefix_tokens > 0 and (
        resolved_prefix_query_precision == "int8" and "int8" in video_phases
    )
    prefix_phase = (
        _resolve_sm120_prefix_phase(video_phases, prefix_kv_precision) if prefix_tokens else None
    )
    phases = (
        video_phases if prefix_phase is None else _active_sm120_phases(video_phases, prefix_phase)
    )

    frames, height, width = video_shape
    current_stream = torch.cuda.current_stream(q_bshd.device)
    prefix_stream = _sm120_prefix_stream(q_bshd.device)
    prefix_output = q_bshd[:, :0].permute(0, 2, 1, 3)
    if prefix_tokens and not native_prefix_int8:
        prefix_stream.wait_stream(current_stream)
        if profile_nvtx:
            torch.cuda.nvtx.range_push("anemoi.dense_prefix")
        with torch.cuda.stream(prefix_stream):
            prefix_output = _dense_prefix_sdpa(q_bshd[:, :prefix_tokens], k_bshd, v_bshd)
        if profile_nvtx:
            torch.cuda.nvtx.range_pop()
    if profile_nvtx:
        torch.cuda.nvtx.range_push("anemoi.peripheral.prepare")
    layout = materialize_ragged_2d_layout(
        q_bshd.device,
        frames=frames,
        height=height,
        width=width,
        logical_block=query_block_size,
        enable_anchors=enable_anchors,
    )
    video_blocks = layout.counts.numel()
    batch = q_bshd.size(0)
    heads = q_bshd.size(2)
    video_counts = layout.counts.view(1, video_blocks).expand(batch, -1).contiguous()
    global_scales = _nvfp4_global_scales(q_bshd.device, nvfp4_scales) if "nvfp4" in phases else None
    nv_operands = None
    int8_operands = None
    mxfp8_operands = None
    prefix_int8_operands = None
    q_max_pool = None
    k_max_pool = None
    raw_query = q_bshd.permute(0, 2, 1, 3)
    raw_key = k_bshd.permute(0, 2, 1, 3)
    raw_value = v_bshd.permute(0, 2, 1, 3)
    if any(phase in phases for phase in ("nvfp4", "int8", "mxfp8")):
        (
            q_pool,
            k_pool,
            query_fp16,
            key_fp16,
            value_fp16,
            nv_operands,
            mxfp8_operands,
            int8_operands,
            prefix_int8_operands,
            q_max_pool,
            k_max_pool,
        ) = _prepare_h3_sm120_inputs(
            raw_query,
            raw_key,
            raw_value,
            video_token_indices=layout.indices,
            video_slot_valid=layout.slot_valid,
            video_valid_counts=layout.counts,
            prefix_tokens=prefix_tokens,
            query_block_size=query_block_size,
            phases=phases,
            has_prefix_query_int8=native_prefix_int8,
            has_maxpool=maxpool_weight != 0.0,
            global_scales=global_scales,
        )
    else:
        query_fp16, key_fp16, value_fp16 = pack_h3_k64_qkv(
            raw_query,
            raw_key,
            raw_value,
            layout.indices,
            layout.slot_valid,
            prefix_tokens,
        )
    _warmup_sync("SM120 QKV preparation", q_bshd.device)
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
        .expand(batch, -1)
        .contiguous()
    )
    valid_k = (
        _expand_q128_valid_k(prefix_counts, video_counts)
        if query_block_size == 128
        else torch.cat((prefix_counts, video_counts), dim=1).contiguous()
    )
    key_video = key_fp16[:, :, prefix_capacity:]

    if not any(phase in phases for phase in ("nvfp4", "int8", "mxfp8")):
        q_pool = _pool_query(query_fp16, video_counts, query_block_size)
        k_pool = _pool_query(key_video, video_counts, query_block_size)
        if maxpool_weight != 0.0:
            q_max_pool = _pool_query(
                query_fp16, video_counts, query_block_size, maximum=True
            )
            k_max_pool = _pool_query(
                key_video, video_counts, query_block_size, maximum=True
            )
    if profile_nvtx:
        torch.cuda.nvtx.range_pop()
    if native_prefix_int8:
        prefix_stream.wait_stream(current_stream)
        prefix_function = (
            sm120_q64_prefix_int8_attention
            if query_block_size == 64
            else sm120_q128_prefix_int8_attention
        )
        if profile_nvtx:
            torch.cuda.nvtx.range_push("anemoi.dense_prefix")
        with torch.cuda.stream(prefix_stream):
            prefix_output = prefix_function(
                prefix_int8_operands[0],
                prefix_int8_operands[1],
                int8_operands,
                valid_k,
                prefix_tokens,
            )
        if profile_nvtx:
            torch.cuda.nvtx.range_pop()
    if profile_nvtx:
        torch.cuda.nvtx.range_push("anemoi.peripheral.draft_route_materialize")
    q_second = None
    k_second = None
    if diag_jensen:
        denominator = video_counts.to(torch.float32).view(1, 1, video_blocks, 1)
        q_second = (
            query_fp16.view(1, heads, video_blocks, query_block_size, -1)
            .float()
            .square_()
            .sum(dim=3)
            .div_(denominator)
            .to(torch.float16)
            .contiguous()
        )
        k_second = (
            key_video.view(1, heads, video_blocks, query_block_size, -1)
            .float()
            .square_()
            .sum(dim=3)
            .div_(denominator)
            .to(torch.float16)
            .contiguous()
        )
    if diag_jensen:
        probability = draft_probability(q_pool, k_pool, q_second=q_second, k_second=k_second)
    elif draftmap_proxy in ("k_tail_r1", "k_tail_r2"):
        k_tail_probability = (
            sm120_h3_k_tail_r1_probability
            if draftmap_proxy == "k_tail_r1"
            else sm120_h3_k_tail_r2_probability
        )
        probability = k_tail_probability(
            q_pool,
            k_pool,
            key_fp16,
            layout.counts,
            prefix_blocks,
        )
    else:
        if maxpool_weight == 0.0:
            probability = sm120_h3_draft_probability(q_pool, k_pool)
        else:
            probability = sm120_h3_draft_probability(
                q_pool,
                k_pool,
                q_max_pool,
                k_max_pool,
                maxpool_weight=maxpool_weight,
            )
    route = _route_sm120_probability(
        probability,
        layout.anchors,
        layout.anchor_ids,
        anchor_count=layout.anchor_count,
        sparsity_ratio=float(sparsity_ratio),
        ratios=ratios,
    )
    block_ids, nv_counts, middle_counts, fp16_counts = sm120_h3_materialize_route(
        route.block_ids,
        route.nvfp4_counts,
        route.fp8_counts,
        route.fp16_counts,
        query_block_size=query_block_size,
        prefix_blocks=prefix_blocks,
        prefix_phase={
            "nvfp4": 0,
            "int8": 1,
            "mxfp8": 1,
            "fp16": 2,
        }[prefix_phase or phases[0]],
        prefix_first=prefix_phase is None or prefix_phase == phases[0],
        has_fp16="fp16" in phases,
    )
    _warmup_sync("SM120 DraftMap routing", q_bshd.device)

    if profile_nvtx:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("anemoi.mpa_mainloop")
    video_output, _ = _run_sm120_phases(
        query_block_size=query_block_size,
        ratios=ratios,
        prefix_phase=prefix_phase,
        query_fp16=query_fp16,
        key_fp16=key_fp16,
        value_fp16=value_fp16,
        block_ids=block_ids,
        nvfp4_counts=nv_counts,
        middle_counts=middle_counts,
        fp16_counts=fp16_counts,
        valid_k_counts=valid_k,
        layer=layer,
        fp16_prefix_blocks=prefix_blocks,
        prepared_nv_operands=nv_operands,
        prepared_int8_operands=int8_operands,
        prepared_mxfp8_operands=mxfp8_operands,
        prepared_global_scales=global_scales,
        nvfp4_scales=nvfp4_scales,
    )
    if profile_nvtx:
        torch.cuda.nvtx.range_pop()
    del key_video, q_pool, k_pool, q_max_pool, k_max_pool
    del q_second, k_second, probability, route
    _warmup_sync("Q128 mixed attention", q_bshd.device)
    del query_fp16, key_fp16, value_fp16, block_ids
    if profile_nvtx:
        torch.cuda.nvtx.range_push("anemoi.peripheral.assembly")
    current_stream.wait_stream(prefix_stream)
    del valid_k
    prefix_output.record_stream(current_stream)
    output = assemble_h3_k64_output(
        prefix_output,
        video_output,
        layout.inverse,
        output_dtype=q_bshd.dtype,
    )
    _warmup_sync("Q128 output assembly", q_bshd.device)
    if profile_nvtx:
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(q_bshd.device)
        torch.cuda.nvtx.range_pop()
    return output


def execute_ragged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    prefix_tokens: int,
    video_shape: tuple[int, int, int],
    layer: int,
    query_block_size: int,
    sparsity_ratio: float,
    nvfp4_ratio: float,
    int8_ratio: float,
    fp16_ratio: float,
    prefix_kv_precision: str,
    prefix_query_precision: str,
    maxpool_weight: float = 0.0,
    nvfp4_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
    smooth_k: bool = False,
) -> torch.Tensor:
    """Dispatch the stable routing configuration to the native backend."""

    capability = tuple(torch.cuda.get_device_capability(query.device))
    if capability == (12, 0):
        ratios = (nvfp4_ratio, int8_ratio, 0.0, fp16_ratio)
        phases = _sm120_phase_config(ratios)
        if prefix_tokens and prefix_query_precision == "int8" and "int8" not in phases:
            raise RuntimeError("prefix-query INT8 requires an active INT8 phase")
        return sm120_ragged_h3_attention(
            query,
            key,
            value,
            prefix_tokens=prefix_tokens,
            sparsity_ratio=sparsity_ratio,
            retained_nvfp4_ratio=nvfp4_ratio,
            retained_int8_ratio=int8_ratio,
            retained_mxfp8_ratio=0.0,
            retained_fp16_ratio=fp16_ratio,
            video_shape=video_shape,
            layer=layer,
            query_block_size=query_block_size,
            prefix_kv_precision=prefix_kv_precision,
            prefix_query_precision=prefix_query_precision,
            draftmap_proxy="mean",
            diag_jensen=False,
            maxpool_weight=maxpool_weight,
            enable_anchors=False,
            nvfp4_scales=nvfp4_scales,
        )
    if capability == (8, 9):
        if query_block_size not in (64, 128) or nvfp4_ratio != 0.0:
            raise RuntimeError("SM89 supports Q64/Q128 INT8/FP16 only")
        return sm89_ragged_h3_attention(
            query,
            key,
            value,
            prefix_tokens=prefix_tokens,
            sparsity_ratio=sparsity_ratio,
            retained_int8_ratio=int8_ratio,
            retained_fp16_ratio=fp16_ratio,
            video_shape=video_shape,
            query_block_size=query_block_size,
            prefix_kv_precision=prefix_kv_precision,
            prefix_query_precision=prefix_query_precision,
            smooth_k=smooth_k,
            diag_jensen=False,
            maxpool_weight=maxpool_weight,
            enable_anchors=False,
        )
    raise RuntimeError("Anemoi attention requires SM89 or SM120")


__all__ = [
    "execute_ragged_attention",
    "sm89_ragged_h3_attention",
    "sm120_ragged_h3_attention",
]
