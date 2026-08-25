from __future__ import annotations

import math
from typing import Any


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Sparse attention requires PyTorch") from exc
    return torch


def dense_attention_reference(
    q: Any,
    k: Any,
    v: Any,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> Any:
    torch = _require_torch()
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must have shape [batch, seq, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError("q, k, and v must share batch, heads, and head_dim")

    batch, q_len, num_heads, head_dim = q.shape
    k_len = k.shape[1]
    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    for b_idx in range(batch):
        for h_idx in range(num_heads):
            scores = torch.mm(q[b_idx, :, h_idx, :].float(), k[b_idx, :, h_idx, :].float().t())
            scores = scores * softmax_scale
            if causal:
                causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
                scores = scores.masked_fill(~causal_mask, -float("inf"))
            probs = torch.softmax(scores, dim=-1)
            out[b_idx, :, h_idx, :] = torch.mm(probs, v[b_idx, :, h_idx, :].float()).to(q.dtype)
    return out


def block_sparse_attention_reference(
    q: Any,
    k: Any,
    v: Any,
    block_mask: Any,
    q_block_size: int,
    k_block_size: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    token_mask: Any | None = None,
) -> Any:
    torch = _require_torch()
    if dropout_p != 0.0:
        raise ValueError("The reference sparse attention path is inference-only and does not support dropout")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must have shape [batch, seq, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError("q, k, and v must share batch, heads, and head_dim")

    batch, q_len, num_heads, head_dim = q.shape
    k_len = k.shape[1]
    expected_q_blocks = (q_len + q_block_size - 1) // q_block_size
    expected_k_blocks = (k_len + k_block_size - 1) // k_block_size
    if tuple(block_mask.shape) != (batch, num_heads, expected_q_blocks, expected_k_blocks):
        raise ValueError(
            "block_mask must have shape "
            f"{(batch, num_heads, expected_q_blocks, expected_k_blocks)}, "
            f"got {tuple(block_mask.shape)}"
        )
    if token_mask is not None and tuple(token_mask.shape) != (batch, k_len):
        raise ValueError(f"token_mask must have shape {(batch, k_len)}, got {tuple(token_mask.shape)}")

    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    out = torch.zeros_like(q)
    for b_idx in range(batch):
        for h_idx in range(num_heads):
            q_head = q[b_idx, :, h_idx, :].float()
            k_head = k[b_idx, :, h_idx, :].float()
            v_head = v[b_idx, :, h_idx, :].float()
            for q_block_idx in range(expected_q_blocks):
                q_start = q_block_idx * q_block_size
                q_end = min(q_start + q_block_size, q_len)
                allowed_scores = []
                allowed_values = []
                allowed_positions = []
                q_block = q_head[q_start:q_end, :]

                for k_block_idx in range(expected_k_blocks):
                    if not bool(block_mask[b_idx, h_idx, q_block_idx, k_block_idx].item()):
                        continue
                    k_start = k_block_idx * k_block_size
                    k_end = min(k_start + k_block_size, k_len)
                    scores = torch.mm(q_block, k_head[k_start:k_end, :].t()) * softmax_scale
                    if token_mask is not None:
                        scores = scores.masked_fill(
                            ~token_mask[b_idx, k_start:k_end].view(1, -1), -float("inf")
                        )
                    allowed_scores.append(scores)
                    allowed_values.append(v_head[k_start:k_end, :])
                    allowed_positions.append((k_start, k_end))

                if not allowed_scores:
                    continue

                scores = torch.cat(allowed_scores, dim=-1)
                values = torch.cat(allowed_values, dim=0)

                if causal:
                    cursor = 0
                    row_positions = torch.arange(q_start, q_end, device=q.device).view(-1, 1)
                    for k_start, k_end in allowed_positions:
                        col_positions = torch.arange(k_start, k_end, device=q.device).view(1, -1)
                        invalid = col_positions > row_positions
                        width = k_end - k_start
                        scores[:, cursor : cursor + width] = scores[:, cursor : cursor + width].masked_fill(
                            invalid, -float("inf")
                        )
                        cursor += width

                probs = torch.softmax(scores, dim=-1)
                out[b_idx, q_start:q_end, h_idx, :] = torch.mm(probs, values).to(q.dtype)
    if token_mask is not None and q_len == k_len:
        out = out.masked_fill(~token_mask[:, :, None, None], 0)
    return out


def block_sparse_attention(
    q: Any,
    k: Any,
    v: Any,
    block_mask: Any,
    q_block_size: int,
    k_block_size: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    backend: str = "auto",
    token_mask: Any | None = None,
    dense_q_start_block: int | None = None,
) -> Any:
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"Unknown sparse attention backend '{backend}'")
    if backend in ("auto", "triton") and getattr(q, "is_cuda", False):
        try:
            from anemoi.layers.attention.triton.block_sparse_attention import triton_block_sparse_attention

            return triton_block_sparse_attention(
                q,
                k,
                v,
                block_mask,
                q_block_size=q_block_size,
                k_block_size=k_block_size,
                softmax_scale=softmax_scale,
                causal=causal,
                token_mask=token_mask,
                dense_q_start_block=dense_q_start_block,
            )
        except (ImportError, RuntimeError):
            if backend == "triton":
                raise
    return block_sparse_attention_reference(
        q,
        k,
        v,
        block_mask,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        softmax_scale=softmax_scale,
        causal=causal,
        dropout_p=dropout_p,
        token_mask=token_mask,
    )
