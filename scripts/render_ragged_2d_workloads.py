#!/usr/bin/env python3
"""Render complete ASCII maps for representative ragged 2-D workloads."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anemoi.layers.attention.mpa.ragged_2d import (  # noqa: E402
    _orientation_candidate,
    _partition_cost,
    _transpose_to_raster,
    make_ragged_2d_partition,
)


CAPACITY = 64
OUTPUT = ROOT / "docs/ragged_2d_workload_partitions.md"
_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Workload:
    title: str
    media: str
    latent_frames: tuple[int, ...]
    height: int
    width: int
    derivation: str


WORKLOADS = (
    Workload(
        "Wan2.1 14B T2V, 720p, 81 frames",
        "81 × 720 × 1280",
        (21,),
        45,
        80,
        "Wan VAE stride (4,8,8), then DiT patch (1,2,2): "
        "T=1+(81-1)/4=21, H=720/16=45, W=1280/16=80.",
    ),
    Workload(
        "LTX-2.3 22B T2V, 1080p two-stage: base spatial stage",
        "361 × 544 × 960",
        (46,),
        17,
        30,
        "LTX VAE stride (8,32,32), patch size 1: "
        "T=1+(361-1)/8=46, H=544/32=17, W=960/32=30.",
    ),
    Workload(
        "LTX-2.3 22B T2V, 1080p: x2 spatial / optional x2 temporal stage",
        "361→721 × 1088 × 1920",
        (46, 91),
        34,
        60,
        "The x2 spatial stage uses 34×60. Before temporal upsampling it has "
        "46 latent frames; 721 output frames give T=1+(721-1)/8=91. "
        "Both temporal lengths reuse this same per-frame spatial map.",
    ),
    Workload(
        "Ideogram 4 T2I, square 2K",
        "1 × 2048 × 2048",
        (1,),
        128,
        128,
        "Official patch_size=2 and AE scale=8 give a 16-pixel token stride: "
        "H=W=2048/16=128.",
    ),
    Workload(
        "Bernini-14B V2V editing, official default",
        "81 × 480 × 848",
        (21,),
        30,
        53,
        "Bernini-R uses the Wan2.2 VAE/transformer layout: "
        "T=1+(81-1)/4=21, H=480/16=30, W=848/16=53.",
    ),
    Workload(
        "Wan2.1 1.3B T2V, 480p, 81 frames",
        "81 × 480 × 832",
        (21,),
        30,
        52,
        "Wan VAE stride (4,8,8), then DiT patch (1,2,2): "
        "T=21, H=480/16=30, W=832/16=52.",
    ),
    Workload(
        "MiniMax-H3 720p established-quality workload",
        "124 generated frames × 768 × 1344",
        (37,),
        24,
        42,
        "The local production runner materializes video_shape=(37,24,42); "
        "the final decoded video contains 120 frames. Spatial token stride is 32.",
    ),
)


def _base36(value: int) -> str:
    if value == 0:
        return "0"
    encoded = ""
    while value:
        value, digit = divmod(value, 36)
        encoded = _DIGITS[digit] + encoded
    return encoded


def _legacy_blocks(height: int, width: int) -> tuple[tuple[int, ...], ...]:
    block_count = math.ceil(height * width / CAPACITY)
    horizontal, _ = _orientation_candidate(
        height, width, CAPACITY, block_count
    )
    transposed, _ = _orientation_candidate(
        width, height, CAPACITY, block_count
    )
    vertical = _transpose_to_raster(transposed, height, width)
    return (
        horizontal
        if (_partition_cost(horizontal, width), False)
        <= (_partition_cost(vertical, width), True)
        else vertical
    )


def _perimeter_lower_bound(counts: tuple[int, ...]) -> int:
    return sum(2 * math.ceil(2 * math.sqrt(count)) for count in counts)


def _count_summary(counts: tuple[int, ...]) -> str:
    return " + ".join(
        f"{counts.count(count)}×{count}" for count in sorted(set(counts))
    )


def _partition_sha256(blocks: tuple[tuple[int, ...], ...]) -> str:
    payload = json.dumps(blocks, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _ascii_map(height: int, width: int, token_to_block: tuple[int, ...]) -> str:
    block_count = max(token_to_block) + 1
    id_width = max(2, len(_base36(block_count - 1)))
    row_width = max(3, len(str(height - 1)))
    lines = []
    for row in range(height):
        labels = (
            _base36(token_to_block[row * width + column]).rjust(id_width, "0")
            for column in range(width)
        )
        lines.append(f"r{row:0{row_width}d} | " + " ".join(labels))
    return "\n".join(lines)


def _block_size_legend(counts: tuple[int, ...]) -> str:
    id_width = max(2, len(_base36(len(counts) - 1)))
    groups: dict[int, list[str]] = {}
    for block_id, count in enumerate(counts):
        groups.setdefault(count, []).append(
            _base36(block_id).rjust(id_width, "0")
        )
    return "; ".join(
        f"{count} tokens: {', '.join(ids)}"
        for count, ids in sorted(groups.items())
    )


def _render() -> str:
    rows = []
    sections = []
    for workload in WORKLOADS:
        partition = make_ragged_2d_partition(
            workload.height,
            workload.width,
            CAPACITY,
            include_adjacency=False,
        )
        legacy = _legacy_blocks(workload.height, workload.width)
        perimeter = int(_partition_cost(partition.blocks, workload.width)[0])
        legacy_perimeter = int(_partition_cost(legacy, workload.width)[0])
        lower_bound = _perimeter_lower_bound(partition.counts)
        latent_frame_summary = "→".join(map(str, workload.latent_frames))
        total_block_values = tuple(
            frames * partition.block_count for frames in workload.latent_frames
        )
        total_blocks = "→".join(map(str, total_block_values))
        rows.append(
            "| "
            + " | ".join(
                (
                    workload.title,
                    workload.media,
                    f"{latent_frame_summary}×{workload.height}×{workload.width}",
                    str(workload.height * workload.width),
                    f"{partition.block_count}/frame; {total_blocks} total",
                    _count_summary(partition.counts),
                    f"{legacy_perimeter}→{perimeter}",
                    str(lower_bound),
                )
            )
            + " |"
        )
        sections.extend(
            (
                f"## {workload.title}",
                "",
                workload.derivation,
                "",
                f"- Spatial tokens/frame: `{workload.height * workload.width}`",
                f"- Blocks/frame: `{partition.block_count}`; all latent frames: "
                f"`{total_blocks}`",
                f"- Exact block counts: `{_count_summary(partition.counts)}`",
                f"- Per-frame perimeter: legacy `{legacy_perimeter}` → selected "
                f"`{perimeter}`; independent per-block lower bound `{lower_bound}`",
                f"- Partition SHA256: `{_partition_sha256(partition.blocks)}`",
                f"- ID sizes: {_block_size_legend(partition.counts)}",
                "",
                "```text",
                _ascii_map(
                    workload.height,
                    workload.width,
                    partition.token_to_block,
                ),
                "```",
                "",
            )
        )

    header = [
        "# Compact ragged 2-D partitions for representative workloads",
        "",
        "This file is generated by `scripts/render_ragged_2d_workloads.py` from "
        "the production `make_ragged_2d_partition` implementation at physical "
        "capacity `C=64`. Every ASCII cell is one transformer video/image token; "
        "the two-character base-36 value is its block ID. Rows and columns are "
        "complete—there are no omitted cells. For video, the same spatial map is "
        "materialized independently on every latent frame.",
        "",
        "The optimization contract is strict: `B=ceil(HW/64)`, every block has "
        "one of the two globally balanced `q/q+1` sizes, every block is "
        "four-connected, and the selected candidate lexicographically minimizes "
        "perimeter, bounding-box waste, spatial moment, and aspect error over the "
        "compact and legacy candidate families.",
        "",
        "## Latent-shape evidence",
        "",
        "- Wan2.1 defaults to 81 frames at 1280×720 and exposes a DiT patch "
        "size `(1,2,2)` in its official "
        "[generator](https://github.com/Wan-Video/Wan2.1/blob/main/generate.py) "
        "and [model](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py). "
        "Its pipeline combines that patch with the `(4,8,8)` VAE stride.",
        "- Bernini's official CLI defaults to 81 frames at 848×480, and its "
        "renderer uses the Wan2.2 base: "
        "[Bernini CLI](https://github.com/bytedance/Bernini/blob/main/bernini/cli.py), "
        "[Bernini repository](https://github.com/bytedance/Bernini).",
        "- LTX-2 defines VAE factors 8 temporal and 32 spatial, uses video "
        "patch size 1, and its two-stage pipeline uses a spatial upscaler: "
        "[latent preprocessing](https://github.com/Lightricks/LTX-2/blob/main/"
        "packages/ltx-trainer/scripts/process_videos.py), "
        "[patchifier](https://github.com/Lightricks/LTX-2/blob/main/packages/"
        "ltx-core/src/ltx_core/components/patchifiers.py), "
        "[official README](https://github.com/Lightricks/LTX-2).",
        "- Ideogram 4's official pipeline sets `patch_size=2`, "
        "`ae_scale_factor=8`, and computes `grid_h/w = pixels/16`; its official "
        "highest-quality example is 2048×2048: "
        "[pipeline](https://github.com/ideogram-oss/ideogram4/blob/main/src/"
        "ideogram4/pipeline_ideogram4.py), "
        "[README](https://github.com/ideogram-oss/ideogram4).",
        "- MiniMax-H3 is verified directly against the local production "
        "[runner](../anemoi/models/minimax_h3/runner.py) and "
        "[reproduction contract](../anemoi/models/minimax_h3/REPRODUCTION.md).",
        "",
        "## Summary",
        "",
        "| Workload | Pixel/media shape | DiT video shape T×H×W | Tokens/frame | "
        "Blocks | Exact sizes/frame | Perimeter old→new | Lower bound |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "The perimeter numbers are per spatial frame. The lower bound is the "
        "sum of the independent minimum polyomino perimeter "
        "`2*ceil(2*sqrt(area))` for every required block area; it need not be "
        "simultaneously tileable inside the complete H×W rectangle.",
        "",
    ]
    return "\n".join((*header, *sections))


def main() -> None:
    OUTPUT.write_text(_render(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
