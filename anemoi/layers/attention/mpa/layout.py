"""Device metadata for compact ragged 2-D packing and routing."""

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
    anchor_ids: Any
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
    enable_anchors: bool,
) -> Ragged2DDeviceLayout:
    import torch

    partition = make_ragged_2d_partition(
        height,
        width,
        logical_block,
        include_adjacency=enable_anchors,
    )
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

    anchors = None
    anchor_ids = None
    anchor_count = 0
    if enable_anchors:
        assert partition.adjacency is not None
        total_blocks = frames * blocks_per_frame
        local_pairs = torch.tensor(
            tuple(
                (source, target)
                for source, row in enumerate(partition.adjacency)
                for target, adjacent in enumerate(row)
                if adjacent
            ),
            device=device,
            dtype=torch.int64,
        )
        frame_offsets = (
            torch.arange(frames, device=device, dtype=torch.int64)
            * blocks_per_frame
        ).view(-1, 1)
        sources = frame_offsets + local_pairs[:, 0]
        targets = frame_offsets + local_pairs[:, 1]
        anchor_ids = (
            (sources * total_blocks + targets).reshape(-1).to(torch.int32).contiguous()
        )
        anchors = torch.zeros(
            (total_blocks, total_blocks), device=device, dtype=torch.bool
        )
        anchors.view(-1)[anchor_ids.to(torch.int64)] = True
        anchor_count = anchor_ids.numel()
    return Ragged2DDeviceLayout(
        indices=indices,
        slot_valid=slot_valid,
        counts=counts,
        inverse=inverse,
        anchors=anchors,
        anchor_ids=anchor_ids,
        anchor_count=anchor_count,
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
    enable_anchors: bool = False,
) -> Ragged2DDeviceLayout:
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("ragged MPA layout requires a CUDA device")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    if logical_block not in (64, 128):
        raise ValueError("logical_block must be 64 or 128")
    if type(enable_anchors) is not bool:
        raise TypeError("enable_anchors must be a built-in bool")
    if any(type(value) is not int or value <= 0 for value in (frames, height, width)):
        raise ValueError("video dimensions must be positive integers")
    return _materialize_layout(
        resolved.type,
        index,
        frames,
        height,
        width,
        logical_block,
        enable_anchors,
    )


__all__ = ["Ragged2DDeviceLayout", "materialize_ragged_2d_layout"]
