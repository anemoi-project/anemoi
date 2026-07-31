from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Draft Attention requires PyTorch") from exc
    return torch, F


@dataclass(frozen=True)
class DraftMapConfig:
    latent_h: int
    latent_w: int
    pool_h: int = 8
    pool_w: int = 16
    sparsity_ratio: float = 0.9
    q_chunk_size: int = 64
    k_chunk_size: int = 64
    backend: str = "auto"

    @property
    def keep_ratio(self) -> float:
        return 1.0 - self.sparsity_ratio


@dataclass(frozen=True)
class DraftMask:
    block_mask: Any
    row_lse: Any
    thresholds: Any
    keep_ratio: float
    q_block_size: int
    k_block_size: int
    pooled_q_len: int
    pooled_k_len: int

    @property
    def density(self) -> float:
        return float(self.block_mask.float().mean().item())


def pool_qk_2d(q: Any, k: Any, config: DraftMapConfig) -> tuple[Any, Any]:
    torch, F = _require_torch()
    if q.dim() != 3 or k.dim() != 3:
        raise ValueError("q and k must have shape [visual_len, num_heads, head_dim]")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}")

    visual_len, num_heads, head_dim = q.shape
    frame_tokens = config.latent_h * config.latent_w
    if visual_len % frame_tokens != 0:
        raise ValueError(
            "visual_len must be divisible by latent_h * latent_w, got "
            f"{visual_len} vs {config.latent_h} * {config.latent_w}"
        )

    num_frames = visual_len // frame_tokens
    q_vid = q.view(num_frames, config.latent_h, config.latent_w, num_heads, head_dim)
    k_vid = k.view(num_frames, config.latent_h, config.latent_w, num_heads, head_dim)

    q_vid = q_vid.permute(0, 3, 4, 1, 2).reshape(
        num_frames, num_heads * head_dim, config.latent_h, config.latent_w
    )
    k_vid = k_vid.permute(0, 3, 4, 1, 2).reshape(
        num_frames, num_heads * head_dim, config.latent_h, config.latent_w
    )

    q_pooled = F.avg_pool2d(
        q_vid,
        kernel_size=(config.pool_h, config.pool_w),
        stride=(config.pool_h, config.pool_w),
        ceil_mode=True,
    )
    k_pooled = F.avg_pool2d(
        k_vid,
        kernel_size=(config.pool_h, config.pool_w),
        stride=(config.pool_h, config.pool_w),
        ceil_mode=True,
    )

    pooled_h, pooled_w = q_pooled.shape[-2:]
    pooled_len = num_frames * pooled_h * pooled_w

    def unmerge(x: Any) -> Any:
        x = x.reshape(num_frames, num_heads, head_dim, pooled_h, pooled_w)
        return x.permute(0, 3, 4, 1, 2).reshape(pooled_len, num_heads, head_dim)

    return unmerge(q_pooled).contiguous(), unmerge(k_pooled).contiguous()


def dense_draft_attention_map(q_draft: Any, k_draft: Any, scale: float | None = None) -> Any:
    torch, _ = _require_torch()
    if q_draft.dim() != 3 or k_draft.dim() != 3:
        raise ValueError("q_draft and k_draft must have shape [draft_len, num_heads, head_dim]")
    if q_draft.shape != k_draft.shape:
        raise ValueError(
            f"q_draft and k_draft must have the same shape, got "
            f"{tuple(q_draft.shape)} and {tuple(k_draft.shape)}"
        )
    head_dim = q_draft.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    q_heads = q_draft.permute(1, 0, 2).float().contiguous()
    k_heads = k_draft.permute(1, 0, 2).float().contiguous()
    scores = torch.bmm(q_heads, k_heads.transpose(1, 2)) * scale
    return torch.softmax(scores, dim=-1)


def headwise_topk_mask(attn_map: Any, keep_ratio: float) -> Any:
    torch, _ = _require_torch()
    if attn_map.dim() != 3:
        raise ValueError("attn_map must have shape [num_heads, q_len, k_len]")
    if keep_ratio <= 0.0:
        return torch.zeros_like(attn_map, dtype=torch.bool)
    if keep_ratio >= 1.0:
        return torch.ones_like(attn_map, dtype=torch.bool)

    num_heads, q_len, k_len = attn_map.shape
    keep_count = max(1, int(math.ceil(keep_ratio * q_len * k_len)))
    mask = torch.zeros_like(attn_map, dtype=torch.bool)
    for head in range(num_heads):
        values = torch.topk(attn_map[head].reshape(-1), keep_count, largest=True)[0]
        threshold = values[-1]
        mask[head] = attn_map[head] >= threshold
    return mask


def blockwise_qk_softmax_lse(
    q_heads: Any,
    k_heads: Any,
    scale: float,
    q_chunk_size: int,
    k_chunk_size: int,
    backend: str = "auto",
) -> Any:
    torch, _ = _require_torch()
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"Unknown draft-map backend '{backend}'")
    if backend in ("auto", "triton") and getattr(q_heads, "is_cuda", False):
        try:
            from evg.layers.attention.triton.draft_map import triton_qk_softmax_lse

            return triton_qk_softmax_lse(
                q_heads,
                k_heads,
                scale=scale,
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
            )
        except (ImportError, RuntimeError):
            if backend == "triton":
                raise
    if q_heads.dim() != 3 or k_heads.dim() != 3:
        raise ValueError("q_heads and k_heads must have shape [num_heads, seq_len, head_dim]")
    if q_heads.shape[0] != k_heads.shape[0] or q_heads.shape[2] != k_heads.shape[2]:
        raise ValueError("q_heads and k_heads must share num_heads and head_dim")

    num_heads, q_len, _ = q_heads.shape
    k_len = k_heads.shape[1]
    device = q_heads.device
    row_m = torch.empty((num_heads, q_len), dtype=torch.float32, device=device)
    row_l = torch.empty((num_heads, q_len), dtype=torch.float32, device=device)

    q_heads = q_heads.float()
    k_heads = k_heads.float()

    for q_start in range(0, q_len, q_chunk_size):
        q_end = min(q_start + q_chunk_size, q_len)
        q_block = q_heads[:, q_start:q_end, :]
        m_i = torch.full((num_heads, q_end - q_start), -float("inf"), dtype=torch.float32, device=device)
        l_i = torch.zeros((num_heads, q_end - q_start), dtype=torch.float32, device=device)

        for k_start in range(0, k_len, k_chunk_size):
            k_end = min(k_start + k_chunk_size, k_len)
            k_block = k_heads[:, k_start:k_end, :]
            scores = torch.bmm(q_block, k_block.transpose(1, 2)) * scale
            block_m = scores.max(dim=-1)[0]
            new_m = torch.max(m_i, block_m)
            alpha = torch.exp(m_i - new_m)
            p_sum = torch.exp(scores - new_m.unsqueeze(-1)).sum(dim=-1)
            l_i = l_i * alpha + p_sum
            m_i = new_m

        row_m[:, q_start:q_end] = m_i
        row_l[:, q_start:q_end] = l_i

    return row_m + torch.log(row_l)


def blockwise_qk_log_probs(
    q_heads: Any,
    k_heads: Any,
    row_lse: Any,
    scale: float,
    q_chunk_size: int,
    k_chunk_size: int,
    backend: str = "auto",
) -> Any:
    torch, _ = _require_torch()
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"Unknown draft-map backend '{backend}'")
    if backend in ("auto", "triton") and getattr(q_heads, "is_cuda", False):
        try:
            from evg.layers.attention.triton.draft_map import triton_qk_log_probs

            return triton_qk_log_probs(
                q_heads,
                k_heads,
                row_lse,
                scale=scale,
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
            )
        except (ImportError, RuntimeError):
            if backend == "triton":
                raise

    num_heads, q_len, _ = q_heads.shape
    k_len = k_heads.shape[1]
    log_probs = torch.empty(
        (num_heads, q_len, k_len), dtype=torch.float32, device=q_heads.device
    )
    q_heads = q_heads.float()
    k_heads = k_heads.float()

    for q_start in range(0, q_len, q_chunk_size):
        q_end = min(q_start + q_chunk_size, q_len)
        q_block = q_heads[:, q_start:q_end, :]
        q_lse = row_lse[:, q_start:q_end].unsqueeze(-1)
        for k_start in range(0, k_len, k_chunk_size):
            k_end = min(k_start + k_chunk_size, k_len)
            k_block = k_heads[:, k_start:k_end, :]
            scores = torch.bmm(q_block, k_block.transpose(1, 2)) * scale
            log_probs[:, q_start:q_end, k_start:k_end] = scores - q_lse

    return log_probs


def _global_topk_mask(log_probs: Any, keep_ratio: float) -> tuple[Any, Any]:
    torch, _ = _require_torch()
    if log_probs.dim() != 3:
        raise ValueError("log_probs must have shape [num_heads, q_len, k_len]")
    num_heads, q_len, k_len = log_probs.shape
    if keep_ratio <= 0.0:
        thresholds = torch.full(
            (num_heads,), float("inf"), dtype=torch.float32, device=log_probs.device
        )
        return torch.zeros_like(log_probs, dtype=torch.bool), thresholds
    if keep_ratio >= 1.0:
        thresholds = torch.full(
            (num_heads,), -float("inf"), dtype=torch.float32, device=log_probs.device
        )
        return torch.ones_like(log_probs, dtype=torch.bool), thresholds

    keep_count = max(1, int(math.ceil(keep_ratio * q_len * k_len)))
    top_values = torch.topk(
        log_probs.reshape(num_heads, -1),
        keep_count,
        dim=1,
        largest=True,
        sorted=False,
    ).values
    thresholds = top_values.min(dim=1).values.contiguous()
    mask = log_probs >= thresholds[:, None, None]
    return mask, thresholds


def blockwise_draft_mask(
    q_draft: Any,
    k_draft: Any,
    keep_ratio: float,
    q_chunk_size: int = 64,
    k_chunk_size: int = 64,
    scale: float | None = None,
    backend: str = "auto",
    output_k_len: int | None = None,
) -> DraftMask:
    if q_draft.dim() != 3 or k_draft.dim() != 3:
        raise ValueError("q_draft and k_draft must have shape [draft_len, num_heads, head_dim]")
    if q_draft.shape[1:] != k_draft.shape[1:]:
        raise ValueError("q_draft and k_draft must share num_heads and head_dim")
    if output_k_len is None:
        output_k_len = k_draft.shape[0]
    if not 0 < output_k_len <= k_draft.shape[0]:
        raise ValueError("output_k_len must select a non-empty prefix of k_draft")
    _, num_heads, head_dim = q_draft.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    q_heads = q_draft.permute(1, 0, 2).contiguous()
    k_heads = k_draft.permute(1, 0, 2).contiguous()

    row_lse = blockwise_qk_softmax_lse(
        q_heads, k_heads, scale, q_chunk_size, k_chunk_size, backend=backend
    )
    log_probs = blockwise_qk_log_probs(
        q_heads,
        k_heads,
        row_lse,
        scale,
        q_chunk_size,
        k_chunk_size,
        backend=backend,
    )
    visual_log_probs = log_probs[:, :, :output_k_len]
    mask, thresholds = _global_topk_mask(visual_log_probs, keep_ratio)

    return DraftMask(
        block_mask=mask,
        row_lse=row_lse,
        thresholds=thresholds,
        keep_ratio=keep_ratio,
        q_block_size=q_chunk_size,
        k_block_size=k_chunk_size,
        pooled_q_len=mask.shape[1],
        pooled_k_len=mask.shape[2],
    )


def build_draft_mask(q: Any, k: Any, config: DraftMapConfig) -> DraftMask:
    q_draft, k_draft = pool_qk_2d(q, k, config)
    return blockwise_draft_mask(
        q_draft,
        k_draft,
        keep_ratio=config.keep_ratio,
        q_chunk_size=config.q_chunk_size,
        k_chunk_size=config.k_chunk_size,
        backend=config.backend,
    )
