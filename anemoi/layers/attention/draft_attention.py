from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from anemoi.layers.attention.draft_map import DraftMapConfig, DraftMask, build_draft_mask
from anemoi.layers.attention.sparse_attention import block_sparse_attention, dense_attention_reference


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Draft Attention requires PyTorch") from exc
    return torch


@dataclass(frozen=True)
class DraftAttentionConfig:
    latent_h: int
    latent_w: int
    visual_len: int
    text_len: int = 0
    pool_h: int = 8
    pool_w: int = 16
    sparsity_ratio: float = 0.9
    draft_q_chunk_size: int = 64
    draft_k_chunk_size: int = 64
    sparse_q_block_size: int | None = None
    sparse_k_block_size: int | None = None
    backend: str = "auto"
    profile: bool = False

    @property
    def sequence_len(self) -> int:
        return self.visual_len + self.text_len

    @property
    def keep_ratio(self) -> float:
        return 1.0 - self.sparsity_ratio


@dataclass(frozen=True)
class DraftAttentionDebugInfo:
    draft_density: float
    sequence_density: float
    draft_q_blocks: int
    draft_k_blocks: int
    sequence_q_blocks: int
    sequence_k_blocks: int
    q_block_size: int
    k_block_size: int
    backend: str


def generate_reorg_indices(
    total_len: int,
    part_size: int,
    block_size: int,
    sub_block_size: int,
) -> list[int]:
    if total_len % part_size != 0:
        raise ValueError("total_len must be a multiple of part_size")
    if part_size % block_size != 0:
        raise ValueError("part_size must be a multiple of block_size")
    if block_size % sub_block_size != 0:
        raise ValueError("block_size must be a multiple of sub_block_size")

    num_parts = total_len // part_size
    blocks_per_part = part_size // block_size
    subs_per_block = block_size // sub_block_size
    part_pattern: list[int] = []
    for sub_idx in range(subs_per_block):
        for block_idx in range(blocks_per_part):
            start = block_idx * block_size + sub_idx * sub_block_size
            part_pattern.extend(range(start, start + sub_block_size))

    reorg_idx: list[int] = []
    for part_idx in range(num_parts):
        base = part_idx * part_size
        reorg_idx.extend(base + offset for offset in part_pattern)
    return reorg_idx


def generate_reorg_restore_indices(
    pool_h: int,
    pool_w: int,
    latent_h: int,
    latent_w: int,
    visual_len: int,
    text_len: int = 0,
) -> tuple[list[int], list[int]]:
    part_size = latent_w * pool_h
    block_size = latent_w
    sub_block_size = pool_w
    if latent_h % pool_h != 0:
        raise ValueError("latent_h must be a multiple of pool_h")
    if visual_len % part_size != 0:
        raise ValueError("visual_len must be a multiple of latent_w * pool_h")
    if block_size % sub_block_size != 0:
        raise ValueError("latent_w must be a multiple of pool_w")

    reorg_idx = generate_reorg_indices(visual_len, part_size, block_size, sub_block_size)
    restore_idx = [0] * visual_len
    for new_pos, old_pos in enumerate(reorg_idx):
        restore_idx[old_pos] = new_pos

    if text_len > 0:
        text_indices = list(range(visual_len, visual_len + text_len))
        reorg_idx.extend(text_indices)
        restore_idx.extend(text_indices)
    return reorg_idx, restore_idx


def generate_padded_reorg_layout(
    pool_h: int,
    pool_w: int,
    latent_h: int,
    latent_w: int,
    visual_len: int,
    text_len: int = 0,
) -> tuple[list[int], list[int], list[bool], int]:
    frame_tokens = latent_h * latent_w
    if visual_len % frame_tokens != 0:
        raise ValueError("visual_len must be a multiple of latent_h * latent_w")

    num_frames = visual_len // frame_tokens
    pooled_h = int(math.ceil(latent_h / pool_h))
    pooled_w = int(math.ceil(latent_w / pool_w))
    block_size = pool_h * pool_w
    padded_visual_len = num_frames * pooled_h * pooled_w * block_size

    reorg_idx: list[int] = []
    valid_mask: list[bool] = []
    restore_idx = [0] * (visual_len + text_len)
    reorg_position = 0
    for frame_idx in range(num_frames):
        frame_offset = frame_idx * frame_tokens
        for tile_h_idx in range(pooled_h):
            for tile_w_idx in range(pooled_w):
                for local_h in range(pool_h):
                    row = tile_h_idx * pool_h + local_h
                    for local_w in range(pool_w):
                        col = tile_w_idx * pool_w + local_w
                        is_valid = row < latent_h and col < latent_w
                        old_position = frame_offset + row * latent_w + col if is_valid else 0
                        reorg_idx.append(old_position)
                        valid_mask.append(is_valid)
                        if is_valid:
                            restore_idx[old_position] = reorg_position
                        reorg_position += 1

    for text_idx in range(text_len):
        old_position = visual_len + text_idx
        new_position = padded_visual_len + text_idx
        reorg_idx.append(old_position)
        valid_mask.append(True)
        restore_idx[old_position] = new_position

    return reorg_idx, restore_idx, valid_mask, padded_visual_len


class DraftAttention:
    def __init__(self, config: DraftAttentionConfig):
        self.config = config
        self._reorg_idx: Any | None = None
        self._restore_idx: Any | None = None
        self._layout_valid_mask: Any | None = None
        self._profile_events: dict[str, list[tuple[Any, Any]]] = {
            stage: []
            for stage in ("draft_map", "reorder", "sparse_attention", "restore", "total")
        }
        self._profile_calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def reset_profile(self) -> None:
        for events in self._profile_events.values():
            events.clear()
        self._profile_calls = 0

    def profile_summary(self, synchronize: bool = True) -> dict[str, float | int]:
        torch = _require_torch()
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        totals = {
            stage: sum(start.elapsed_time(end) for start, end in events)
            for stage, events in self._profile_events.items()
        }
        measured_stages = (
            totals["draft_map"]
            + totals["reorder"]
            + totals["sparse_attention"]
            + totals["restore"]
        )
        totals["framework_overhead"] = max(0.0, totals["total"] - measured_stages)
        totals["additional_overhead"] = max(
            0.0, totals["total"] - totals["sparse_attention"]
        )
        totals["calls"] = self._profile_calls
        return totals

    def _profile_start(self, tensor: Any) -> Any | None:
        if not self.config.profile or not getattr(tensor, "is_cuda", False):
            return None
        torch = _require_torch()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _profile_end(self, stage: str, start: Any | None) -> None:
        if start is None:
            return
        torch = _require_torch()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._profile_events[stage].append((start, end))

    def _indices(self, device: Any) -> tuple[Any, Any, Any]:
        torch = _require_torch()
        if (
            self._reorg_idx is None
            or self._restore_idx is None
            or self._layout_valid_mask is None
            or self._reorg_idx.device != device
        ):
            reorg_idx, restore_idx, valid_mask, _ = generate_padded_reorg_layout(
                pool_h=self.config.pool_h,
                pool_w=self.config.pool_w,
                latent_h=self.config.latent_h,
                latent_w=self.config.latent_w,
                visual_len=self.config.visual_len,
                text_len=self.config.text_len,
            )
            self._reorg_idx = torch.tensor(reorg_idx, dtype=torch.long, device=device)
            self._restore_idx = torch.tensor(restore_idx, dtype=torch.long, device=device)
            self._layout_valid_mask = torch.tensor(
                valid_mask, dtype=torch.bool, device=device
            ).unsqueeze(0)
        return self._reorg_idx, self._restore_idx, self._layout_valid_mask

    def build_sequence_block_mask(
        self, q: Any, k: Any, collect_debug: bool = True
    ) -> tuple[Any, DraftAttentionDebugInfo]:
        torch = _require_torch()
        if q.dim() != 4 or k.dim() != 4:
            raise ValueError("q and k must have shape [batch, seq, heads, head_dim]")
        if q.shape != k.shape:
            raise ValueError(f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}")
        batch, input_seq_len, num_heads, _ = q.shape
        if input_seq_len < self.config.sequence_len:
            raise ValueError(
                f"sequence length {input_seq_len} is smaller than configured length {self.config.sequence_len}"
            )
        per_batch_masks: list[Any] = []
        draft_masks: list[DraftMask] = []
        for b_idx in range(batch):
            draft_config = DraftMapConfig(
                latent_h=self.config.latent_h,
                latent_w=self.config.latent_w,
                pool_h=self.config.pool_h,
                pool_w=self.config.pool_w,
                sparsity_ratio=self.config.sparsity_ratio,
                q_chunk_size=self.config.draft_q_chunk_size,
                k_chunk_size=self.config.draft_k_chunk_size,
                backend=self.config.backend,
            )
            draft_mask = build_draft_mask(
                q[b_idx, : self.config.visual_len, :, :],
                k[b_idx, : self.config.visual_len, :, :],
                draft_config,
            )
            draft_masks.append(draft_mask)

            q_block_size = self.config.sparse_q_block_size
            if q_block_size is None:
                q_block_size = self.config.pool_h * self.config.pool_w
            k_block_size = self.config.sparse_k_block_size
            if k_block_size is None:
                k_block_size = self.config.pool_h * self.config.pool_w

            padded_visual_q_len = draft_mask.pooled_q_len * q_block_size
            padded_visual_k_len = draft_mask.pooled_k_len * k_block_size
            sparse_q_len = padded_visual_q_len + self.config.text_len
            sparse_k_len = padded_visual_k_len + self.config.text_len
            seq_q_blocks = int(math.ceil(float(sparse_q_len) / q_block_size))
            seq_k_blocks = int(math.ceil(float(sparse_k_len) / k_block_size))
            seq_mask = torch.ones(
                (num_heads, seq_q_blocks, seq_k_blocks), dtype=torch.bool, device=q.device
            )
            seq_mask[
                :,
                : draft_mask.block_mask.shape[1],
                : draft_mask.block_mask.shape[2],
            ] = draft_mask.block_mask
            per_batch_masks.append(seq_mask)

        block_mask = torch.stack(per_batch_masks, dim=0)
        first_draft_mask = draft_masks[0]
        q_block_size = self.config.sparse_q_block_size
        if q_block_size is None:
            q_block_size = self.config.pool_h * self.config.pool_w
        k_block_size = self.config.sparse_k_block_size
        if k_block_size is None:
            k_block_size = self.config.pool_h * self.config.pool_w

        debug = DraftAttentionDebugInfo(
            draft_density=(
                float(sum(mask.density for mask in draft_masks) / len(draft_masks))
                if collect_debug
                else float("nan")
            ),
            sequence_density=(
                float(block_mask.float().mean().item()) if collect_debug else float("nan")
            ),
            draft_q_blocks=first_draft_mask.pooled_q_len,
            draft_k_blocks=first_draft_mask.pooled_k_len,
            sequence_q_blocks=block_mask.shape[-2],
            sequence_k_blocks=block_mask.shape[-1],
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            backend=self.config.backend,
        )
        return block_mask, debug

    def forward(
        self,
        q: Any,
        k: Any,
        v: Any,
        causal: bool = False,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        batch_size: int | None = None,
        max_seqlen_q: int | None = None,
        attn_mask: Any | None = None,
        return_debug: bool = False,
    ) -> Any:
        torch = _require_torch()
        original_rank = q.dim()
        if original_rank == 3:
            if batch_size is None or max_seqlen_q is None:
                raise ValueError("batch_size and max_seqlen_q are required for [total_seq, heads, dim] inputs")
            q_bshd = q.view(batch_size, max_seqlen_q, q.shape[-2], q.shape[-1])
            k_bshd = k.view(batch_size, max_seqlen_q, k.shape[-2], k.shape[-1])
            v_bshd = v.view(batch_size, max_seqlen_q, v.shape[-2], v.shape[-1])
        elif original_rank == 4:
            q_bshd, k_bshd, v_bshd = q, k, v
            batch_size = q.shape[0]
            max_seqlen_q = q.shape[1]
        else:
            raise ValueError("q, k, and v must have shape [batch, seq, heads, dim] or [total_seq, heads, dim]")

        if self.config.sparsity_ratio <= 0.0:
            out = dense_attention_reference(
                q_bshd, k_bshd, v_bshd, softmax_scale=softmax_scale, causal=causal
            )
            debug = DraftAttentionDebugInfo(
                draft_density=1.0,
                sequence_density=1.0,
                draft_q_blocks=0,
                draft_k_blocks=0,
                sequence_q_blocks=0,
                sequence_k_blocks=0,
                q_block_size=0,
                k_block_size=0,
                backend="dense-reference",
            )
            if original_rank == 3:
                out = out.reshape(batch_size * max_seqlen_q, out.shape[-2], out.shape[-1])
            return (out, debug) if return_debug else out

        if max_seqlen_q != self.config.sequence_len:
            raise ValueError(
                f"max_seqlen_q must equal visual_len + text_len for Draft Attention, "
                f"got {max_seqlen_q} vs {self.config.sequence_len}"
            )
        if attn_mask is not None and tuple(attn_mask.shape) != (
            batch_size,
            max_seqlen_q,
        ):
            raise ValueError(
                f"attn_mask must have shape {(batch_size, max_seqlen_q)}, "
                f"got {tuple(attn_mask.shape)}"
            )

        total_start = self._profile_start(q_bshd)
        draft_map_start = self._profile_start(q_bshd)
        block_mask, debug = self.build_sequence_block_mask(
            q_bshd,
            k_bshd,
            collect_debug=return_debug,
        )
        self._profile_end("draft_map", draft_map_start)

        reorder_start = self._profile_start(q_bshd)
        reorg_idx, restore_idx, layout_valid_mask = self._indices(q_bshd.device)

        q_reorg = q_bshd.index_select(1, reorg_idx)
        k_reorg = k_bshd.index_select(1, reorg_idx)
        v_reorg = v_bshd.index_select(1, reorg_idx)
        self._profile_end("reorder", reorder_start)
        token_mask = layout_valid_mask.expand(batch_size, -1)
        if attn_mask is not None:
            token_mask = token_mask & attn_mask.to(
                device=q_bshd.device, dtype=torch.bool
            ).index_select(1, reorg_idx)
        sparse_attention_start = self._profile_start(q_bshd)
        out = block_sparse_attention(
            q_reorg,
            k_reorg,
            v_reorg,
            block_mask=block_mask,
            q_block_size=debug.q_block_size,
            k_block_size=debug.k_block_size,
            softmax_scale=softmax_scale,
            causal=causal,
            dropout_p=dropout_p,
            backend=self.config.backend,
            token_mask=token_mask,
            dense_q_start_block=debug.draft_q_blocks,
        )
        self._profile_end("sparse_attention", sparse_attention_start)

        restore_start = self._profile_start(out)
        out = out.index_select(1, restore_idx)
        self._profile_end("restore", restore_start)
        self._profile_end("total", total_start)
        if total_start is not None:
            self._profile_calls += 1
        if original_rank == 3:
            out = out.reshape(batch_size * max_seqlen_q, out.shape[-2], out.shape[-1])
        return (out, debug) if return_debug else out
