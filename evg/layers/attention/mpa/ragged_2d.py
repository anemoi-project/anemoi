"""Request-static stripe-DP partitioning for ragged 2-D attention.

The production partition has one contract: cover an arbitrary positive
``height x width`` grid with exactly ``ceil(height * width / capacity)``
connected blocks, each containing at most ``capacity`` real tokens. No
padding token becomes part of the logical partition.

The dynamic program considers complete row bands in both orientations.  A
band is walked column-wise in a serpentine order and split into balanced
connected intervals.  The winning exact-block-count partition minimizes, in
order, discrete perimeter, bounding-box waste, spatial moment, and square
aspect error.  Partition construction is host-only and cached per request
geometry; every request still executes the same physical K64 CUDA kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Ragged2DPartition:
    """Immutable metadata for one exact-cover rectangular partition."""

    height: int
    width: int
    capacity: int
    blocks: tuple[tuple[int, ...], ...]
    token_to_block: tuple[int, ...]
    adjacency: tuple[tuple[bool, ...], ...]

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def counts(self) -> tuple[int, ...]:
        """Return the real-token denominator of every physical block."""

        return tuple(len(block) for block in self.blocks)


_Cost = tuple[float, float, float, float, int]


def _balanced_segment_sizes(tokens: int, blocks: int) -> tuple[int, ...]:
    base, larger = divmod(tokens, blocks)
    return (base + 1,) * larger + (base,) * (blocks - larger)


def _band_serpentine_blocks(
    height: int,
    width: int,
    block_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Split a complete row band into balanced connected intervals."""

    path = tuple(
        row * width + column
        for column in range(width)
        for row in (range(height) if column % 2 == 0 else range(height - 1, -1, -1))
    )
    blocks: list[tuple[int, ...]] = []
    offset = 0
    for size in _balanced_segment_sizes(height * width, block_count):
        blocks.append(path[offset : offset + size])
        offset += size
    return tuple(blocks)


def _block_shape_terms(
    block: tuple[int, ...],
    width: int,
) -> tuple[int, int, float, float]:
    cells = {divmod(token, width) for token in block}
    rows = [row for row, _ in cells]
    columns = [column for _, column in cells]
    box_height = max(rows) - min(rows) + 1
    box_width = max(columns) - min(columns) + 1
    perimeter = sum(
        neighbor not in cells
        for row, column in cells
        for neighbor in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        )
    )
    centroid_row = sum(rows) / len(rows)
    centroid_column = sum(columns) / len(columns)
    moment = sum(
        (row - centroid_row) ** 2 + (column - centroid_column) ** 2 for row, column in cells
    )
    aspect_error = abs(math.log(box_width / box_height))
    return (
        perimeter,
        box_height * box_width - len(block),
        moment,
        aspect_error,
    )


def _partition_cost(
    blocks: tuple[tuple[int, ...], ...],
    width: int,
) -> _Cost:
    terms = [_block_shape_terms(block, width) for block in blocks]
    return (
        float(sum(term[0] for term in terms)),
        float(sum(term[1] for term in terms)),
        sum(term[2] for term in terms),
        sum(term[3] for term in terms),
        1,
    )


def _add_cost(lhs: _Cost, rhs: _Cost) -> _Cost:
    return tuple(left + right for left, right in zip(lhs, rhs, strict=True))  # type: ignore[return-value]


def _orientation_candidate(
    height: int,
    width: int,
    capacity: int,
    target_blocks: int,
) -> tuple[tuple[tuple[int, ...], ...], _Cost]:
    """Return the best horizontal-band partition with an exact block count."""

    zero: _Cost = (0.0, 0.0, 0.0, 0.0, 0)
    states: dict[
        tuple[int, int],
        tuple[_Cost, tuple[tuple[int, int], ...]],
    ] = {(0, 0): (zero, ())}
    band_cache: dict[
        tuple[int, int],
        tuple[tuple[tuple[int, ...], ...], _Cost],
    ] = {}

    for row_offset in range(height):
        for blocks_used in range(target_blocks + 1):
            state = states.get((row_offset, blocks_used))
            if state is None:
                continue
            old_cost, old_bands = state
            for band_height in range(1, height - row_offset + 1):
                minimum_band_blocks = math.ceil(band_height * width / capacity)
                remaining_rows = height - row_offset - band_height
                minimum_remaining_blocks = math.ceil(remaining_rows * width / capacity)
                maximum_band_blocks = min(
                    band_height * width,
                    target_blocks - blocks_used - minimum_remaining_blocks,
                )
                for band_blocks in range(minimum_band_blocks, maximum_band_blocks + 1):
                    cache_key = (band_height, band_blocks)
                    cached = band_cache.get(cache_key)
                    if cached is None:
                        local_blocks = _band_serpentine_blocks(band_height, width, band_blocks)
                        cached = (
                            local_blocks,
                            _partition_cost(local_blocks, width),
                        )
                        band_cache[cache_key] = cached
                    candidate = (
                        _add_cost(old_cost, cached[1]),
                        old_bands + ((band_height, band_blocks),),
                    )
                    key = (row_offset + band_height, blocks_used + band_blocks)
                    incumbent = states.get(key)
                    if incumbent is None or candidate < incumbent:
                        states[key] = candidate

    final = states.get((height, target_blocks))
    if final is None:
        raise RuntimeError("stripe DP failed to find the capacity-minimum cover")

    score, bands = final
    blocks: list[tuple[int, ...]] = []
    row_offset = 0
    for band_height, band_blocks in bands:
        for block in band_cache[(band_height, band_blocks)][0]:
            blocks.append(
                tuple((token // width + row_offset) * width + token % width for token in block)
            )
        row_offset += band_height
    return tuple(blocks), score


def _transpose_to_raster(
    blocks: tuple[tuple[int, ...], ...],
    height: int,
    width: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            transposed_column * width + transposed_row
            for token in block
            for transposed_row, transposed_column in (divmod(token, height),)
        )
        for block in blocks
    )


def _partition_adjacency(
    height: int,
    width: int,
    token_to_block: list[int],
    block_count: int,
) -> tuple[tuple[bool, ...], ...]:
    adjacency = [[False] * block_count for _ in range(block_count)]
    for block_id in range(block_count):
        adjacency[block_id][block_id] = True
    for row in range(height):
        for column in range(width):
            token = row * width + column
            source = token_to_block[token]
            if column + 1 < width:
                target = token_to_block[token + 1]
                adjacency[source][target] = True
                adjacency[target][source] = True
            if row + 1 < height:
                target = token_to_block[token + width]
                adjacency[source][target] = True
                adjacency[target][source] = True
    return tuple(tuple(row) for row in adjacency)


@lru_cache(maxsize=128)
def make_ragged_2d_partition(
    height: int,
    width: int,
    capacity: int = 64,
) -> Ragged2DPartition:
    """Return the frozen stripe-compact partition for an arbitrary grid."""

    if any(type(value) is not int or value <= 0 for value in (height, width)):
        raise ValueError("height and width must be positive integers")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive integer")

    block_count = math.ceil(height * width / capacity)
    if height == 1 or width == 1:
        # A connected subset of a one-dimensional grid is an interval.  The
        # moment-minimizing exact cover therefore has balanced consecutive
        # interval sizes; this is the same candidate selected by the general
        # two-orientation DP without its quadratic degenerate-axis search.
        selected = _band_serpentine_blocks(1, height * width, block_count)
    else:
        horizontal_blocks, _ = _orientation_candidate(height, width, capacity, block_count)
        transposed_blocks, _ = _orientation_candidate(width, height, capacity, block_count)
        vertical_blocks = _transpose_to_raster(transposed_blocks, height, width)
        # Score in the original coordinate system.  The Boolean makes exact
        # ties deterministic and favors the ordinary construction.
        selected = (
            horizontal_blocks
            if (_partition_cost(horizontal_blocks, width), False)
            <= (_partition_cost(vertical_blocks, width), True)
            else vertical_blocks
        )
    # Serpentine order proves connectivity; canonical raster order makes
    # rectangular blocks byte-equivalent to the historical aligned layout.
    blocks = tuple(tuple(sorted(block)) for block in selected)

    token_to_block = [-1] * (height * width)
    for block_id, block in enumerate(blocks):
        if not block or len(block) > capacity:
            raise RuntimeError("ragged block violates physical K capacity")
        for token in block:
            if not 0 <= token < height * width:
                raise RuntimeError("ragged block contains an invalid token")
            if token_to_block[token] != -1:
                raise RuntimeError("ragged partition assigned a token twice")
            token_to_block[token] = block_id
    if any(block_id < 0 for block_id in token_to_block):
        raise RuntimeError("ragged partition did not cover the complete grid")
    if len(blocks) != block_count:
        raise RuntimeError("ragged partition did not reach the capacity lower bound")

    return Ragged2DPartition(
        height=height,
        width=width,
        capacity=capacity,
        blocks=blocks,
        token_to_block=tuple(token_to_block),
        adjacency=_partition_adjacency(height, width, token_to_block, block_count),
    )


__all__ = ["Ragged2DPartition", "make_ragged_2d_partition"]
