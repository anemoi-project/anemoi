"""Frozen geometry policy for the MiniMax-H3 SM89 mainline.

The route owns a logical two-dimensional tile.  The Ada attention mainloop
owns an independent physical K64 tile.  Keeping those choices separate lets
the request-level geometry selector avoid spatial padding without compiling a
different attention kernel for every image shape.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


SM89_PHYSICAL_K_TOKENS = 64

# Quality-validated, single-stage logical shapes.  Do not grow this list from
# a padding heuristic alone: every new topology changes DraftMap centroids and
# legal-neighbour relationships and therefore needs an end-to-end quality run.
SM89_VALIDATED_LOGICAL_TILES = ((8, 8), (8, 7), (7, 8))


@dataclass(frozen=True)
class SM89Adaptive2DPlan:
    """One request-level logical/physical tiling decision."""

    logical_tile: tuple[int, int]
    physical_k_tokens: int
    blocks_per_frame: int
    spatial_padding_tokens_per_frame: int
    physical_inactive_lanes_per_frame: int


def select_sm89_adaptive_2d_plan(
    video_shape: tuple[int, int, int],
) -> SM89Adaptive2DPlan:
    """Choose a validated logical tile while retaining one physical K64 path.

    The primary cost is the number of physical K64 blocks.  Among equal-cost
    candidates, less logical spatial padding wins.  The last tie follows the
    explicit validated order above, making the decision deterministic and
    preventing an unvalidated orientation change.
    """

    if (
        not isinstance(video_shape, tuple)
        or len(video_shape) != 3
        or any(type(value) is not int or value <= 0 for value in video_shape)
    ):
        raise ValueError("video_shape must contain three positive integers")
    _, height, width = video_shape

    choices: list[tuple[int, int, int, tuple[int, int]]] = []
    for order, tile in enumerate(SM89_VALIDATED_LOGICAL_TILES):
        tile_height, tile_width = tile
        blocks = math.ceil(height / tile_height) * math.ceil(width / tile_width)
        spatial_capacity = blocks * tile_height * tile_width
        spatial_padding = spatial_capacity - height * width
        choices.append((blocks, spatial_padding, order, tile))
    blocks, spatial_padding, _, tile = min(choices)
    physical_capacity = blocks * SM89_PHYSICAL_K_TOKENS
    return SM89Adaptive2DPlan(
        logical_tile=tile,
        physical_k_tokens=SM89_PHYSICAL_K_TOKENS,
        blocks_per_frame=blocks,
        spatial_padding_tokens_per_frame=spatial_padding,
        physical_inactive_lanes_per_frame=physical_capacity - height * width,
    )


__all__ = [
    "SM89Adaptive2DPlan",
    "SM89_PHYSICAL_K_TOKENS",
    "SM89_VALIDATED_LOGICAL_TILES",
    "select_sm89_adaptive_2d_plan",
]
