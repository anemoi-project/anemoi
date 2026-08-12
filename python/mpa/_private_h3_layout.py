"""Recognize the released packed MiniMax-H3 Ulysses Q/K/V views."""

from __future__ import annotations

import torch


_SEQUENCE = 38_247
_PREFIX = 951
_HEADS = 14
_HEAD_DIM = 128
_PROJECTION_STRIDE = 3 * _HEAD_DIM
_TOKEN_STRIDE = _HEADS * _PROJECTION_STRIDE


def _storage_contains(tensor: torch.Tensor) -> bool:
    last = tensor.storage_offset()
    for size, stride in zip(tensor.shape, tensor.stride()):
        if size <= 0 or stride < 0:
            return False
        last += (size - 1) * stride
    return (last + 1) * tensor.element_size() <= tensor.untyped_storage().nbytes()


def _is_exact_h3_packed_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    video_shape: tuple[int, int, int],
    prefix_tokens: int,
) -> bool:
    """Return whether Q/K/V are the validated packed 4-way Ulysses split."""

    if video_shape != (37, 24, 42) or prefix_tokens != _PREFIX:
        return False
    tensors = (q, k, v)
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and compiler.is_compiling():
        return False
    if any(
        not isinstance(tensor, torch.Tensor)
        or tensor.shape != (_SEQUENCE, _HEADS, _HEAD_DIM)
        or tensor.stride() != (_TOKEN_STRIDE, _PROJECTION_STRIDE, 1)
        or tensor.dtype != torch.bfloat16
        or tensor.device.type != "cuda"
        for tensor in tensors
    ):
        return False
    if k.device != q.device or v.device != q.device:
        return False
    if torch.cuda.get_device_capability(q.device) != (8, 9):
        return False
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        return False
    try:
        if len({tensor.untyped_storage().data_ptr() for tensor in tensors}) != 1:
            return False
    except RuntimeError:
        return False
    q_offset = q.storage_offset()
    if (
        q_offset % _PROJECTION_STRIDE
        or k.storage_offset() != q_offset + _HEAD_DIM
        or v.storage_offset() != q_offset + 2 * _HEAD_DIM
        or any(tensor.data_ptr() % 16 for tensor in tensors)
    ):
        return False
    return all(_storage_contains(tensor) for tensor in tensors)


__all__ = []
