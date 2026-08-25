"""Package-private SM120 Q128 three-phase backend."""

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
def _load_prepare_nvfp4() -> Callable[..., tuple[torch.Tensor, ...]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator("prepare_q128_nvfp4")


@lru_cache(maxsize=1)
def _load_stack() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q128_nv_int8_fp16_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_nvfp4() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q128_nvfp4_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_fp16() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q128_fp16_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_int8() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q128_int8_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_prefix_int8() -> Callable[..., torch.Tensor]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q128_prefix_int8_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_mxfp8() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q128_mxfp8_attention_forward"
    )


@lru_cache(maxsize=1)
def _load_nv_mxfp8() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    import_native_extension("sm120_q64")
    return resolve_mixed_attention_operator(
        "sm120_q128_nv_mx_fp16_attention_forward"
    )


def prepare_q128_nvfp4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_global_scale: torch.Tensor,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return _load_prepare_nvfp4()(
        query,
        key,
        value,
        q_global_scale,
        k_global_scale,
        v_global_scale,
    )


def sm120_q128_prefix_int8_attention(
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


def sm120_q128_nv_int8_fp16_attention(
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
    active_fp16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_stack()(
        *nv_operands,
        *int8_operands,
        query_fp16,
        key_fp16,
        value_fp16,
        block_ids,
        nvfp4_counts,
        int8_counts,
        fp16_counts,
        valid_k_counts,
        *global_scales,
        fp16_prefix_blocks,
        1.0 / math.sqrt(query_fp16.size(-1)),
        active_fp16,
    )


def sm120_q128_nvfp4_fp16_attention(
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
    active_fp16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_nvfp4()(
        *nv_operands,
        query_fp16,
        key_fp16,
        value_fp16,
        block_ids,
        nvfp4_counts,
        fp16_counts,
        valid_k_counts,
        *global_scales,
        fp16_prefix_blocks,
        1.0 / math.sqrt(query_fp16.size(-1)),
        active_fp16,
    )


def sm120_q128_int8_fp16_attention(
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
    active_fp16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    q8, q8_scale, k8, k8_scale, v8, v8_scale = int8_operands
    return _load_int8()(
        q8,
        k8,
        v8,
        query_fp16,
        key_fp16,
        value_fp16,
        block_ids,
        int8_counts,
        fp16_counts,
        q8_scale,
        k8_scale,
        v8_scale,
        valid_k_counts,
        fp16_prefix_blocks,
        1.0 / math.sqrt(query_fp16.size(-1)),
        active_fp16,
    )


def sm120_q128_mxfp8_attention(
    mxfp8_operands: tuple[torch.Tensor, ...],
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    block_ids: torch.Tensor,
    mxfp8_counts: torch.Tensor,
    fp16_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
    *,
    fp16_prefix_blocks: int,
    active_fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_mxfp8()(
        *mxfp8_operands, query_fp16, key_fp16, value_fp16,
        block_ids, mxfp8_counts, fp16_counts, valid_k_counts,
        fp16_prefix_blocks, 1.0 / math.sqrt(query_fp16.size(-1)), active_fp16,
    )


def sm120_q128_nv_mxfp8_fp16_attention(
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


def sm120_q128_fp16_attention(
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    value_fp16: torch.Tensor,
    block_ids: torch.Tensor,
    block_counts: torch.Tensor,
    valid_k_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_fp16()(
        query_fp16,
        key_fp16,
        value_fp16,
        block_ids,
        block_counts,
        valid_k_counts,
        1.0 / math.sqrt(query_fp16.size(-1)),
    )


__all__ = [
    "prepare_q128_nvfp4",
    "sm120_q128_fp16_attention",
    "sm120_q128_int8_fp16_attention",
    "sm120_q128_mxfp8_attention",
    "sm120_q128_nv_int8_fp16_attention",
    "sm120_q128_nv_mxfp8_fp16_attention",
    "sm120_q128_nvfp4_fp16_attention",
]
