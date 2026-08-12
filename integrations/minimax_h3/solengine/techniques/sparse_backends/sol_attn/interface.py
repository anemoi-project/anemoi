"""Triton Sol-Attn interface used by the RTX 4090 comparison candidate."""

from __future__ import annotations

import torch


BLOCK_SIZE = 64


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    thresh_type: str,
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> tuple[int, int]:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q/K/V must share [B,T,H,128]")
    if q.shape[1] == 0 or q.shape[3] != 128:
        raise ValueError("Sol-Attn requires T > 0 and head_dim=128")
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        raise TypeError("Sol-Attn Q/K/V must be BF16")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("Q/K/V must share one CUDA device")
    if not all(tensor.is_contiguous() for tensor in (q, k, v)):
        raise ValueError("Q/K/V must be contiguous BTHD tensors")
    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    if type(sink_tokens) is not int or not 0 <= sink_tokens <= q.shape[1]:
        raise ValueError("sink_tokens must be an integer in [0,T]")
    if sink_start is not None:
        if type(sink_start) is not int or not 0 <= sink_start <= q.shape[1]:
            raise ValueError("sink_start must be an integer in [0,T] or None")
        if sink_start + sink_tokens > q.shape[1]:
            raise ValueError("sink_start + sink_tokens exceeds T")
    capability = tuple(torch.cuda.get_device_capability(q.device))
    if capability != (8, 9):
        raise RuntimeError(f"the bundled Sol comparison is pinned to SM89, got {capability}")
    return capability


def _sink_block_range(
    tokens: int,
    sink_start: int | None,
    sink_tokens: int,
) -> tuple[int, int]:
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if not sink_tokens:
        return blocks, blocks
    start = tokens - sink_tokens if sink_start is None else sink_start
    return (
        start // BLOCK_SIZE,
        (start + sink_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    kv_splits: int = 1,
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> torch.Tensor:
    """Compute noncausal Sol-Attn for contiguous BF16 BTHD tensors."""

    _validate_inputs(q, k, v, thresh_type, sink_tokens, sink_start)
    if kv_splits != 1:
        raise ValueError("the SM89 Triton path requires kv_splits=1")
    from .triton_ref import sol_attn as triton_sol_attn

    return triton_sol_attn(
        q,
        k,
        v,
        scale=q.shape[-1] ** -0.5 if scale is None else float(scale),
        tau=float(tau),
        thresh_type=thresh_type,
        sink_tokens=sink_tokens,
        sink_start=sink_start,
    )


__all__ = ["sol_attn"]
