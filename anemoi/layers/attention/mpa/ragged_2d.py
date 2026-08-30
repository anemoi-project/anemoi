"""Request-static compact partitioning for ragged 2-D attention.

The production partition covers an arbitrary positive ``height x width`` grid
with exactly ``ceil(height * width / capacity)`` connected blocks. Every block
contains one of the two globally balanced real-token counts and no padding
token becomes part of the logical partition.

The compact candidate first groups blocks into near-square ragged bands, then
walks every band on the perpendicular axis.  Its one-cell seams absorb the
global ``q``/``q+1`` mass constraint without stretching complete blocks across
an entire stripe.  A legacy full-band candidate remains in the search set, so
the selected exact cover can never regress its lexicographic shape objective:
discrete perimeter, bounding-box waste, spatial moment, then square aspect
error.  Construction is host-only and cached per request geometry; every
request still executes the same physical CUDA kernel.
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
    adjacency: tuple[tuple[bool, ...], ...] | None

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
    return tuple(
        round(left + right, 12) if index < 4 else left + right
        for index, (left, right) in enumerate(zip(lhs, rhs, strict=True))
    )  # type: ignore[return-value]


def _orientation_candidate(
    height: int,
    width: int,
    capacity: int,
    target_blocks: int,
) -> tuple[tuple[tuple[int, ...], ...], _Cost]:
    """Return the best horizontal-band partition with an exact block count."""

    small_block_size, large_blocks = divmod(height * width, target_blocks)
    if small_block_size + bool(large_blocks) > capacity:
        raise RuntimeError("balanced block size exceeds physical capacity")
    zero: _Cost = (0.0, 0.0, 0.0, 0.0, 0)
    states: dict[
        tuple[int, int],
        tuple[_Cost, int, tuple[tuple[int, int], ...]],
    ] = {(0, 0): (zero, 0, ())}
    band_cache: dict[
        tuple[int, int],
        tuple[tuple[tuple[int, ...], ...], _Cost],
    ] = {}

    for row_offset in range(height):
        for blocks_used in range(target_blocks + 1):
            state = states.get((row_offset, blocks_used))
            if state is None:
                continue
            old_cost, old_center_cost, old_bands = state
            for band_height in range(1, height - row_offset + 1):
                band_tokens = band_height * width
                minimum_band_blocks = math.ceil(
                    band_tokens / (small_block_size + 1)
                )
                maximum_band_blocks = min(
                    band_tokens // small_block_size,
                    target_blocks - blocks_used,
                )
                remaining_rows = height - row_offset - band_height
                for band_blocks in range(minimum_band_blocks, maximum_band_blocks + 1):
                    band_large = band_tokens - band_blocks * small_block_size
                    if not 0 <= band_large <= band_blocks:
                        continue
                    remaining_blocks = target_blocks - blocks_used - band_blocks
                    remaining_tokens = remaining_rows * width
                    remaining_large = (
                        remaining_tokens - remaining_blocks * small_block_size
                    )
                    if not 0 <= remaining_large <= remaining_blocks:
                        continue
                    cache_key = (band_height, band_blocks)
                    cached = band_cache.get(cache_key)
                    if cached is None:
                        local_blocks = _band_serpentine_blocks(band_height, width, band_blocks)
                        cached = (
                            local_blocks,
                            _partition_cost(local_blocks, width),
                        )
                        band_cache[cache_key] = cached
                    next_row = row_offset + band_height
                    candidate = (
                        _add_cost(old_cost, cached[1]),
                        old_center_cost
                        + (abs(2 * next_row - height) if next_row < height else 0),
                        old_bands + ((band_height, band_blocks),),
                    )
                    key = (next_row, blocks_used + band_blocks)
                    incumbent = states.get(key)
                    if incumbent is None or candidate < incumbent:
                        states[key] = candidate

    final = states.get((height, target_blocks))
    if final is None:
        raise RuntimeError("stripe DP failed to find the capacity-minimum cover")

    score, _, bands = final
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


def _balanced_binary_patterns(length: int, ones: int) -> tuple[tuple[int, ...], ...]:
    """Return a small deterministic set of balanced zero/one arrangements."""

    if not 0 <= ones <= length:
        raise ValueError("ones must be within the binary-pattern length")
    if ones == 0:
        return ((0,) * length,)
    if ones == length:
        return ((1,) * length,)

    even = tuple(
        (index + 1) * ones // length - index * ones // length
        for index in range(length)
    )
    patterns = {
        (1,) * ones + (0,) * (length - ones),
        (0,) * (length - ones) + (1,) * ones,
        even,
        tuple(reversed(even)),
    }
    # Keeping this set constant-sized bounds host construction independently
    # of the number of CUDA blocks while covering clustered and even seams.
    return tuple(sorted(patterns))


def _block_is_connected(block: tuple[int, ...], width: int) -> bool:
    """Return whether raster tokens form one four-connected component."""

    remaining = set(block)
    frontier = [remaining.pop()]
    while frontier:
        token = frontier.pop()
        row, column = divmod(token, width)
        neighbors = []
        if row:
            neighbors.append(token - width)
        if column:
            neighbors.append(token - 1)
        if column + 1 < width:
            neighbors.append(token + 1)
        neighbors.append(token + width)
        for neighbor in neighbors:
            if neighbor in remaining:
                remaining.remove(neighbor)
                frontier.append(neighbor)
    return not remaining


def _nested_serpentine_blocks(
    height: int,
    width: int,
    sizes: tuple[int, ...],
    band_block_counts: tuple[int, ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Split ragged horizontal bands by their perpendicular compact walks."""

    outer_path = tuple(
        row * width + column
        for row in range(height)
        for column in (
            range(width) if row % 2 == 0 else range(width - 1, -1, -1)
        )
    )
    blocks: list[tuple[int, ...]] = []
    token_offset = 0
    block_offset = 0
    for band_blocks in band_block_counts:
        band_sizes = sizes[block_offset : block_offset + band_blocks]
        band_tokens = sum(band_sizes)
        band_cells = set(outer_path[token_offset : token_offset + band_tokens])
        rows_by_column: list[list[int]] = [[] for _ in range(width)]
        for token in band_cells:
            row, column = divmod(token, width)
            rows_by_column[column].append(row)
        for rows in rows_by_column:
            rows.sort()
        best_band: tuple[
            _Cost,
            bool,
            int,
            tuple[tuple[int, ...], ...],
        ] | None = None
        for reverse_columns in (False, True):
            columns = (
                range(width - 1, -1, -1) if reverse_columns else range(width)
            )
            for phase in (0, 1):
                inner_path = tuple(
                    row * width + column
                    for column_index, column in enumerate(columns)
                    for row in (
                        rows_by_column[column]
                        if (column_index + phase) % 2 == 0
                        else reversed(rows_by_column[column])
                    )
                )
                trial: list[tuple[int, ...]] = []
                offset = 0
                for size in band_sizes:
                    block = inner_path[offset : offset + size]
                    if not _block_is_connected(block, width):
                        break
                    trial.append(block)
                    offset += size
                if len(trial) != band_blocks:
                    continue
                trial_blocks = tuple(trial)
                candidate = (
                    _partition_cost(trial_blocks, width),
                    reverse_columns,
                    phase,
                    trial_blocks,
                )
                if best_band is None or candidate < best_band:
                    best_band = candidate
        if best_band is None:
            return None
        blocks.extend(best_band[3])
        token_offset += band_tokens
        block_offset += band_blocks

    if token_offset != height * width or block_offset != len(sizes):
        raise RuntimeError("compact band construction did not consume the exact grid")
    return tuple(blocks)


def _compact_candidate(
    height: int,
    width: int,
    capacity: int,
    target_blocks: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Return the best lower-bound-guided two-level compact candidate."""

    small_size, large_blocks = divmod(height * width, target_blocks)
    if small_size + bool(large_blocks) > capacity:
        raise RuntimeError("balanced block size exceeds physical capacity")

    best: tuple[
        _Cost,
        bool,
        int,
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, ...], ...],
    ] | None = None
    for transposed in (False, True):
        oriented_height, oriented_width = (
            (width, height) if transposed else (height, width)
        )
        ideal_bands = math.sqrt(
            target_blocks * oriented_height / oriented_width
        )
        band_counts = {
            max(1, min(oriented_height, target_blocks, math.floor(ideal_bands))),
            max(1, min(oriented_height, target_blocks, math.ceil(ideal_bands))),
        }
        for band_count in sorted(band_counts):
            small_band, large_bands = divmod(target_blocks, band_count)
            for band_pattern in _balanced_binary_patterns(
                band_count, large_bands
            ):
                band_block_counts = tuple(
                    small_band + bit for bit in band_pattern
                )
                for size_pattern in _balanced_binary_patterns(
                    target_blocks, large_blocks
                ):
                    sizes = tuple(small_size + bit for bit in size_pattern)
                    oriented_blocks = _nested_serpentine_blocks(
                        oriented_height,
                        oriented_width,
                        sizes,
                        band_block_counts,
                    )
                    if oriented_blocks is None:
                        continue
                    blocks = (
                        _transpose_to_raster(oriented_blocks, height, width)
                        if transposed
                        else oriented_blocks
                    )
                    candidate = (
                        _partition_cost(blocks, width),
                        transposed,
                        band_count,
                        band_block_counts,
                        sizes,
                        blocks,
                    )
                    if best is None or candidate < best:
                        best = candidate
    return None if best is None else best[5]


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
    *,
    include_adjacency: bool = True,
) -> Ragged2DPartition:
    """Return the frozen compact partition for an arbitrary grid."""

    if any(type(value) is not int or value <= 0 for value in (height, width)):
        raise ValueError("height and width must be positive integers")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if type(include_adjacency) is not bool:
        raise TypeError("include_adjacency must be a built-in bool")

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
        stripe_selected = (
            horizontal_blocks
            if (_partition_cost(horizontal_blocks, width), False)
            <= (_partition_cost(vertical_blocks, width), True)
            else vertical_blocks
        )
        compact = _compact_candidate(height, width, capacity, block_count)
        # Keep the proven full-band construction as a fallback and as the
        # deterministic winner of exact ties.  The new search therefore only
        # changes layouts when it improves the complete shape objective.
        selected = (
            compact
            if compact is not None
            and _partition_cost(compact, width)
            < _partition_cost(stripe_selected, width)
            else stripe_selected
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
        adjacency=(
            _partition_adjacency(height, width, token_to_block, block_count)
            if include_adjacency
            else None
        ),
    )


__all__ = ["Ragged2DPartition", "make_ragged_2d_partition"]
