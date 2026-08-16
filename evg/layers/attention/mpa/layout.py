"""Device metadata for stripe-compact ragged 2-D packing and routing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .ragged_2d import make_ragged_2d_partition


@dataclass(frozen=True)
class Ragged2DDeviceLayout:
    indices: Any
    slot_valid: Any
    counts: Any
    inverse: Any
    anchors: Any
    anchor_count: int
    blocks_per_frame: int
    logical_block: int


@lru_cache(maxsize=64)
def _materialize_layout(
    device_type: str,
    device_index: int,
    frames: int,
    height: int,
    width: int,
    logical_block: int,
) -> Ragged2DDeviceLayout:
    import torch

    partition = make_ragged_2d_partition(height, width, logical_block)
    device = torch.device(device_type, device_index)
    blocks_per_frame = partition.block_count
    local = torch.tensor(
        tuple(block + (0,) * (logical_block - len(block)) for block in partition.blocks),
        device=device,
        dtype=torch.int64,
    )
    counts = (
        torch.tensor(partition.counts, device=device, dtype=torch.int32).repeat(frames).contiguous()
    )
    frame_offsets = (
        torch.arange(frames, device=device, dtype=torch.int64) * (height * width)
    ).view(frames, 1, 1)
    indices = local.view(1, blocks_per_frame, logical_block) + frame_offsets
    slot_valid = torch.arange(logical_block, device=device).view(1, logical_block) < counts.view(
        -1, 1
    )
    indices.masked_fill_(~slot_valid.view(frames, blocks_per_frame, logical_block), 0)
    indices = indices.reshape(-1).contiguous()
    slot_valid = slot_valid.reshape(-1).contiguous()

    tokens = frames * height * width
    inverse = torch.empty(tokens, device=device, dtype=torch.int64)
    physical_slots = torch.arange(indices.numel(), device=device, dtype=torch.int64)
    inverse.scatter_(0, indices[slot_valid], physical_slots[slot_valid])

    local_adjacency = torch.tensor(partition.adjacency, device=device, dtype=torch.bool)
    total_blocks = frames * blocks_per_frame
    block_ids = torch.arange(total_blocks, device=device, dtype=torch.int64)
    block_frames = torch.div(block_ids, blocks_per_frame, rounding_mode="floor")
    spatial_ids = block_ids.remainder(blocks_per_frame)
    anchors = (
        (block_frames[:, None] == block_frames[None, :])
        & local_adjacency[spatial_ids[:, None], spatial_ids[None, :]]
    ).contiguous()
    return Ragged2DDeviceLayout(
        indices=indices,
        slot_valid=slot_valid,
        counts=counts,
        inverse=inverse,
        anchors=anchors,
        anchor_count=frames * sum(sum(row) for row in partition.adjacency),
        blocks_per_frame=blocks_per_frame,
        logical_block=logical_block,
    )


def materialize_ragged_2d_layout(
    device: Any,
    *,
    frames: int,
    height: int,
    width: int,
    logical_block: int,
) -> Ragged2DDeviceLayout:
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("ragged MPA layout requires a CUDA device")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    if logical_block not in (64, 128):
        raise ValueError("logical_block must be 64 or 128")
    if any(type(value) is not int or value <= 0 for value in (frames, height, width)):
        raise ValueError("video dimensions must be positive integers")
    return _materialize_layout(resolved.type, index, frames, height, width, logical_block)


__all__ = ["Ragged2DDeviceLayout", "materialize_ragged_2d_layout"]
