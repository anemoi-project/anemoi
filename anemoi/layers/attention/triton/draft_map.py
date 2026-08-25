from __future__ import annotations

import math
from typing import Any

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency
    torch = None
    triton = None
    tl = None


if triton is not None:  # pragma: no cover - exercised on CUDA/Triton machines

    @triton.jit
    def _qk_lse_kernel(
        q_ptr,
        k_ptr,
        lse_ptr,
        q_len: tl.constexpr,
        k_len: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q_offsets = (pid_h * q_len + offs_m[:, None]) * head_dim + offs_d[None, :]
        q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
        q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)

        for start_n in range(0, k_len, BLOCK_N):
            cols = start_n + offs_n
            k_offsets = (pid_h * k_len + cols[None, :]) * head_dim + offs_d[:, None]
            k_mask = (cols[None, :] < k_len) & (offs_d[:, None] < head_dim)
            k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
            scores = tl.dot(q, k) * scale
            scores = tl.where(
                (offs_m[:, None] < q_len) & (cols[None, :] < k_len),
                scores,
                -float("inf"),
            )
            block_m = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_m)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        lse = m_i + tl.log(l_i)
        tl.store(lse_ptr + pid_h * q_len + offs_m, lse, mask=offs_m < q_len)

    @triton.jit
    def _qk_log_probs_kernel(
        q_ptr,
        k_ptr,
        lse_ptr,
        out_ptr,
        q_len: tl.constexpr,
        k_len: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_h = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)

        q_offsets = (pid_h * q_len + rows[:, None]) * head_dim + dims[None, :]
        k_offsets = (pid_h * k_len + cols[None, :]) * head_dim + dims[:, None]
        q = tl.load(
            q_ptr + q_offsets,
            mask=(rows[:, None] < q_len) & (dims[None, :] < head_dim),
            other=0.0,
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=(cols[None, :] < k_len) & (dims[:, None] < head_dim),
            other=0.0,
        )
        row_lse = tl.load(
            lse_ptr + pid_h * q_len + rows,
            mask=rows < q_len,
            other=0.0,
        )
        log_probs = tl.dot(q, k) * scale - row_lse[:, None]
        output_offsets = (pid_h * q_len + rows[:, None]) * k_len + cols[None, :]
        tl.store(
            out_ptr + output_offsets,
            log_probs,
            mask=(rows[:, None] < q_len) & (cols[None, :] < k_len),
        )


def _next_power_of_2(value: int) -> int:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    return int(triton.next_power_of_2(value))


def triton_qk_softmax_lse(
    q_heads: Any,
    k_heads: Any,
    scale: float | None = None,
    q_chunk_size: int = 64,
    k_chunk_size: int = 64,
) -> Any:
    if triton is None or torch is None:
        raise ImportError("Triton is not installed")
    if not q_heads.is_cuda or not k_heads.is_cuda:
        raise RuntimeError("Triton draft-map LSE requires CUDA tensors")
    if q_heads.dim() != 3 or k_heads.dim() != 3:
        raise ValueError("q_heads and k_heads must have shape [heads, seq, head_dim]")
    if q_heads.shape[0] != k_heads.shape[0] or q_heads.shape[2] != k_heads.shape[2]:
        raise ValueError("q_heads and k_heads must share heads and head_dim")

    num_heads, q_len, head_dim = q_heads.shape
    k_len = k_heads.shape[1]
    if head_dim > 256:
        raise RuntimeError("The Triton draft-map LSE kernel supports head_dim <= 256")

    q_heads = q_heads.contiguous()
    k_heads = k_heads.contiguous()
    out = torch.empty((num_heads, q_len), dtype=torch.float32, device=q_heads.device)
    block_d = _next_power_of_2(head_dim)
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    grid = (triton.cdiv(q_len, q_chunk_size), num_heads)
    _qk_lse_kernel[grid](
        q_heads,
        k_heads,
        out,
        q_len,
        k_len,
        head_dim,
        scale,
        BLOCK_M=q_chunk_size,
        BLOCK_N=k_chunk_size,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return out


def triton_qk_log_probs(
    q_heads: Any,
    k_heads: Any,
    row_lse: Any,
    scale: float | None = None,
    q_chunk_size: int = 64,
    k_chunk_size: int = 64,
) -> Any:
    if triton is None or torch is None:
        raise ImportError("Triton is not installed")
    if not q_heads.is_cuda or not k_heads.is_cuda:
        raise RuntimeError("Triton draft-map scores require CUDA tensors")
    if q_heads.dim() != 3 or k_heads.dim() != 3:
        raise ValueError("q_heads and k_heads must have shape [heads, seq, head_dim]")
    if q_heads.shape[0] != k_heads.shape[0] or q_heads.shape[2] != k_heads.shape[2]:
        raise ValueError("q_heads and k_heads must share heads and head_dim")

    num_heads, q_len, head_dim = q_heads.shape
    k_len = k_heads.shape[1]
    if tuple(row_lse.shape) != (num_heads, q_len):
        raise ValueError(
            f"row_lse must have shape {(num_heads, q_len)}, got {tuple(row_lse.shape)}"
        )
    if head_dim > 256:
        raise RuntimeError("The Triton draft-map scores kernel supports head_dim <= 256")

    q_heads = q_heads.contiguous()
    k_heads = k_heads.contiguous()
    row_lse = row_lse.contiguous()
    out = torch.empty(
        (num_heads, q_len, k_len), dtype=torch.float32, device=q_heads.device
    )
    block_d = _next_power_of_2(head_dim)
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    grid = (
        triton.cdiv(q_len, q_chunk_size),
        triton.cdiv(k_len, k_chunk_size),
        num_heads,
    )
    _qk_log_probs_kernel[grid](
        q_heads,
        k_heads,
        row_lse,
        out,
        q_len,
        k_len,
        head_dim,
        scale,
        BLOCK_M=q_chunk_size,
        BLOCK_N=k_chunk_size,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return out
