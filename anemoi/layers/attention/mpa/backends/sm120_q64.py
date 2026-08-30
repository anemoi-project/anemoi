"""Package-private SM120 Q64xK64 precision-phase backend."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import torch

from anemoi.layers.attention.mpa.build_identity import (
    import_native_extension,
    resolve_mixed_attention_operator,
)


@lru_cache(maxsize=1)
def _load_ops() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q64_fp16_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_mxfp8_op() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q64_mxfp8_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_prepare_nvfp4() -> Callable[..., tuple[torch.Tensor, ...]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("prepare_q64_nvfp4")


@lru_cache(maxsize=1)
def _load_prepare_mxfp8() -> Callable[..., tuple[torch.Tensor, ...]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("prepare_mxfp8")


@lru_cache(maxsize=1)
def _load_h3_preparation() -> Callable[..., tuple[torch.Tensor, ...]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("prepare_h3_sm120_operands")


@lru_cache(maxsize=1)
def _load_h3_draft() -> Callable[..., torch.Tensor]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("sm120_h3_draft_probability")


@lru_cache(maxsize=1)
def _load_h3_k_tail_r1() -> Callable[..., torch.Tensor]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("sm120_h3_k_tail_r1_probability")


@lru_cache(maxsize=1)
def _load_h3_k_tail_r2() -> Callable[..., torch.Tensor]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("sm120_h3_k_tail_r2_probability")


@lru_cache(maxsize=1)
def _load_h3_route() -> Callable[..., tuple[torch.Tensor, ...]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("sm120_h3_route_precision")


@lru_cache(maxsize=1)
def _load_h3_route_materialization() -> Callable[..., tuple[torch.Tensor, ...]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("sm120_h3_materialize_route")


@lru_cache(maxsize=1)
def _load_int8() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("sm120_q64_int8_attention_forward")


@lru_cache(maxsize=1)
def _load_prefix_int8() -> Callable[..., torch.Tensor]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q64_prefix_int8_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_nvfp4() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("sm120_q64_nvfp4_attention_forward")


@lru_cache(maxsize=1)
def _load_nv_int8() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q64_nv_int8_fp16_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_nv_mxfp8() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q64_nv_mx_fp16_attention_forward"
    )


def prepare_q64_nvfp4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_global_scale: torch.Tensor,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return _load_prepare_nvfp4()(
        query, key, value, q_global_scale, k_global_scale, v_global_scale
    )


def prepare_mxfp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return _load_prepare_mxfp8()(query, key, value)


def prepare_h3_sm120_operands(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    video_token_indices: torch.Tensor,
    video_slot_valid: torch.Tensor,
    video_valid_counts: torch.Tensor,
    *,
    prefix_tokens: int,
    query_block_size: int,
    has_nvfp4: bool,
    has_int8: bool,
    has_mxfp8: bool,
    has_fp16: bool,
    has_prefix_query_int8: bool,
    has_maxpool: bool,
    global_scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, ...]:
    scales = global_scales or (None, None, None)
    return _load_h3_preparation()(
        query,
        key,
        value,
        video_token_indices,
        video_slot_valid,
        video_valid_counts,
        prefix_tokens,
        query_block_size,
        has_nvfp4,
        has_int8,
        has_mxfp8,
        has_fp16,
        has_prefix_query_int8,
        has_maxpool,
        *scales,
    )


def sm120_h3_draft_probability(
    q_pool: torch.Tensor,
    k_pool: torch.Tensor,
    q_max_pool: torch.Tensor | None = None,
    k_max_pool: torch.Tensor | None = None,
    maxpool_weight: float = 0.0,
) -> torch.Tensor:
    operation = _load_h3_draft()
    if maxpool_weight == 0.0:
        return operation(q_pool, k_pool)
    return operation(q_pool, k_pool, q_max_pool, k_max_pool, maxpool_weight)


def sm120_h3_k_tail_r1_probability(
    q_pool: torch.Tensor,
    k_pool: torch.Tensor,
    packed_k: torch.Tensor,
    valid_counts: torch.Tensor,
    prefix_blocks: int,
) -> torch.Tensor:
    return _load_h3_k_tail_r1()(
        q_pool, k_pool, packed_k, valid_counts, prefix_blocks
    )


def sm120_h3_k_tail_r2_probability(
    q_pool: torch.Tensor,
    k_pool: torch.Tensor,
    packed_k: torch.Tensor,
    valid_counts: torch.Tensor,
    prefix_blocks: int,
) -> torch.Tensor:
    return _load_h3_k_tail_r2()(
        q_pool, k_pool, packed_k, valid_counts, prefix_blocks
    )


def sm120_h3_route_precision(
    probability: torch.Tensor,
    n16: int,
    n8: int,
    n4: int,
    anchors: torch.Tensor | None = None,
    anchor_ids: torch.Tensor | None = None,
    anchor_count: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _load_h3_route()(
        probability, n16, n8, n4, anchors, anchor_ids, anchor_count
    )


def sm120_h3_materialize_route(
    logical_ids: torch.Tensor,
    nvfp4_counts: torch.Tensor,
    middle_counts: torch.Tensor,
    fp16_counts: torch.Tensor,
    *,
    query_block_size: int,
    prefix_blocks: int,
    prefix_phase: int,
    prefix_first: bool,
    has_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _load_h3_route_materialization()(
        logical_ids,
        nvfp4_counts,
        middle_counts,
        fp16_counts,
        query_block_size,
        prefix_blocks,
        prefix_phase,
        prefix_first,
        has_fp16,
    )


def sm120_q64_fp16_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_ids: torch.Tensor,
    block_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_ops()(
        query,
        key,
        value,
        block_ids,
        block_counts,
        valid_k_counts,
        1.0 / math.sqrt(query.size(-1)),
    )


def sm120_q64_prefix_int8_attention(
    prefix_q8: torch.Tensor,
    prefix_q_scale: torch.Tensor,
    shared_int8_operands: tuple[torch.Tensor, ...],
    valid_k_counts: torch.Tensor,
    prefix_tokens: int,
) -> torch.Tensor:
    _, _, k8, k_scale, v8, v_scale = shared_int8_operands
    return _load_prefix_int8()(
        prefix_q8,
        k8,
        v8,
        prefix_q_scale,
        k_scale,
        v_scale,
        valid_k_counts,
        prefix_tokens,
        1.0 / math.sqrt(128),
    )


def sm120_q64_mxfp8_attention(
    q_mxfp8: torch.Tensor,
    q_mxfp8_scale: torch.Tensor,
    k_mxfp8: torch.Tensor,
    k_mxfp8_scale: torch.Tensor,
    v_mxfp8: torch.Tensor,
    v_mxfp8_scale: torch.Tensor,
    q_fp16: torch.Tensor,
    k_fp16: torch.Tensor,
    v_fp16: torch.Tensor,
    block_ids: torch.Tensor,
    mxfp8_block_counts: torch.Tensor,
    fp16_block_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
    *,
    fp16_prefix_blocks: int = 0,
    active_fp16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_mxfp8_op()(
        q_mxfp8,
        q_mxfp8_scale,
        k_mxfp8,
        k_mxfp8_scale,
        v_mxfp8,
        v_mxfp8_scale,
        q_fp16,
        k_fp16,
        v_fp16,
        block_ids,
        mxfp8_block_counts,
        fp16_block_counts,
        valid_k_counts,
        fp16_prefix_blocks,
        1.0 / math.sqrt(q_fp16.size(-1)),
        active_fp16,
    )


def sm120_q64_int8_fp16_attention(
    int8_operands: tuple[torch.Tensor, ...],
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    block_ids: torch.Tensor,
    int8_counts: torch.Tensor,
    fp16_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
    *,
    fp16_prefix_blocks: int,
    active_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q8, q8_scale, k8, k8_scale, v8, v8_scale = int8_operands
    return _load_int8()(
        q8, k8, v8, query_fp16, key_fp16, value_fp16,
        block_ids, int8_counts, fp16_counts,
        q8_scale, k8_scale, v8_scale, valid_k_counts,
        fp16_prefix_blocks, 1.0 / math.sqrt(query_fp16.size(-1)), active_fp16,
    )


def sm120_q64_nvfp4_fp16_attention(
    nv_operands: tuple[torch.Tensor, ...],
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    block_ids: torch.Tensor,
    nvfp4_counts: torch.Tensor,
    fp16_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
    global_scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    fp16_prefix_blocks: int,
    active_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_nvfp4()(
        *nv_operands, query_fp16, key_fp16, value_fp16,
        block_ids, nvfp4_counts, fp16_counts, valid_k_counts, *global_scales,
        fp16_prefix_blocks, 1.0 / math.sqrt(query_fp16.size(-1)), active_fp16,
    )


def sm120_q64_nv_int8_fp16_attention(
    nv_operands: tuple[torch.Tensor, ...],
    int8_operands: tuple[torch.Tensor, ...],
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    block_ids: torch.Tensor,
    nvfp4_counts: torch.Tensor,
    int8_counts: torch.Tensor,
    fp16_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
    global_scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    fp16_prefix_blocks: int,
    active_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_nv_int8()(
        *nv_operands, *int8_operands, query_fp16, key_fp16, value_fp16,
        block_ids, nvfp4_counts, int8_counts, fp16_counts, valid_k_counts,
        *global_scales, fp16_prefix_blocks,
        1.0 / math.sqrt(query_fp16.size(-1)), active_fp16,
    )


def sm120_q64_nv_mxfp8_fp16_attention(
    nv_operands: tuple[torch.Tensor, ...],
    mxfp8_operands: tuple[torch.Tensor, ...],
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    block_ids: torch.Tensor,
    nvfp4_counts: torch.Tensor,
    mxfp8_counts: torch.Tensor,
    fp16_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
    global_scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    fp16_prefix_blocks: int,
    active_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_nv_mxfp8()(
        *nv_operands, *mxfp8_operands, query_fp16, key_fp16, value_fp16,
        block_ids, nvfp4_counts, mxfp8_counts, fp16_counts, valid_k_counts,
        *global_scales, fp16_prefix_blocks,
        1.0 / math.sqrt(query_fp16.size(-1)), active_fp16,
    )


def _kernel_metadata(native_name: str, label: str) -> dict[str, int]:
    module = import_native_extension("sm120_q64")
    values = tuple(getattr(module, native_name)())
    keys = (
        "registers",
        "static_smem_bytes",
        "max_dynamic_smem_bytes",
        "active_ctas_per_sm",
    )
    metadata = dict(zip(keys, map(int, values), strict=True))
    if metadata["registers"] > 168 or metadata["active_ctas_per_sm"] < 3:
        raise RuntimeError(f"SM120 Q64 {label} resource gate failed: {metadata}")
    return metadata


def sm120_q64_fp16_kernel_metadata() -> dict[str, int]:
    return _kernel_metadata("sm120_q64_fp16_kernel_metadata", "FP16")


def sm120_q64_mxfp8_kernel_metadata() -> dict[str, int]:
    return _kernel_metadata("sm120_q64_mxfp8_kernel_metadata", "MXFP8")


__all__ = [
    "prepare_h3_sm120_operands",
    "prepare_mxfp8",
    "prepare_q64_nvfp4",
    "sm120_h3_draft_probability",
    "sm120_h3_k_tail_r1_probability",
    "sm120_h3_k_tail_r2_probability",
    "sm120_h3_materialize_route",
    "sm120_h3_route_precision",
    "sm120_q64_fp16_attention",
    "sm120_q64_fp16_kernel_metadata",
    "sm120_q64_int8_fp16_attention",
    "sm120_q64_mxfp8_attention",
    "sm120_q64_mxfp8_kernel_metadata",
    "sm120_q64_nv_int8_fp16_attention",
    "sm120_q64_nv_mxfp8_fp16_attention",
    "sm120_q64_nvfp4_fp16_attention",
]
