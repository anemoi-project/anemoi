#!/usr/bin/env python3
"""Trend benchmark for the donor-first SM89 H3 preparation boundary."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from statistics import median
import sys
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anemoi.layers.attention.mpa.backends.sm89_k64 import (
    pack_h3_k64_qkv,
    prepare_h3_sm89_int8_operands,
    prepare_k64_fp8_operands,
)


def _measure(operation: Callable[[], object], *, warmup: int, iterations: int) -> float:
    result = None
    for _ in range(warmup):
        result = operation()
    torch.cuda.synchronize()
    if result is not None:
        del result
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        result = operation()
    end.record()
    end.synchronize()
    del result
    return start.elapsed_time(end) / iterations


def _slope(points: list[dict[str, float]]) -> float:
    x = [point["video_tokens"] / 1000.0 for point in points]
    y = [point["milliseconds"] for point in points]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator


def _case(
    *,
    query_block: int,
    blocks: int,
    heads: int,
    prefix_tokens: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    video_tokens = query_block * blocks
    shape = (1, heads, prefix_tokens + video_tokens, 128)
    query = torch.randn(shape, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    indices = torch.arange(video_tokens, device="cuda", dtype=torch.int64)
    valid = torch.ones(video_tokens, device="cuda", dtype=torch.bool)
    counts = torch.full(
        (blocks,), query_block, device="cuda", dtype=torch.int32
    )
    denominator = counts.view(1, 1, blocks, 1)
    prefix_capacity = math.ceil(prefix_tokens / 64) * 64

    def unfused():
        packed_q, packed_k, packed_v = pack_h3_k64_qkv(
            query, key, value, indices, valid, prefix_tokens
        )
        operands = prepare_k64_fp8_operands(
            packed_q, packed_k, packed_v, query_block=query_block
        )
        q_pool = (
            packed_q.view(1, heads, blocks, query_block, 128)
            .sum(3, dtype=torch.float32)
            .div(denominator)
            .half()
        )
        k_pool = (
            packed_k[:, :, prefix_capacity:]
            .view(1, heads, blocks, query_block, 128)
            .sum(3, dtype=torch.float32)
            .div(denominator)
            .half()
        )
        return q_pool, k_pool, packed_q, packed_k, packed_v, *operands

    def fused_no_smooth():
        return prepare_h3_sm89_int8_operands(
            query,
            key,
            value,
            indices,
            valid,
            counts,
            prefix_tokens=prefix_tokens,
            query_block_size=query_block,
            smooth_k=False,
        )

    def fused_smooth():
        return prepare_h3_sm89_int8_operands(
            query,
            key,
            value,
            indices,
            valid,
            counts,
            prefix_tokens=prefix_tokens,
            query_block_size=query_block,
            smooth_k=True,
        )

    def fused_maxpool():
        return prepare_h3_sm89_int8_operands(
            query,
            key,
            value,
            indices,
            valid,
            counts,
            prefix_tokens=prefix_tokens,
            query_block_size=query_block,
            smooth_k=False,
            has_maxpool=True,
        )

    timings: dict[str, float] = {}
    for name, operation in (
        ("unfused", unfused),
        ("fused_no_smooth", fused_no_smooth),
        ("fused_smooth", fused_smooth),
        ("fused_maxpool", fused_maxpool),
    ):
        samples = [
            _measure(operation, warmup=warmup, iterations=iterations)
            for _ in range(repeats)
        ]
        timings[name] = median(samples)
    return {
        "query_block": query_block,
        "blocks": blocks,
        "video_tokens": video_tokens,
        **{
            name: {
                "milliseconds": milliseconds,
                "million_video_tokens_per_second": (
                    video_tokens / milliseconds / 1000.0
                ),
            }
            for name, milliseconds in timings.items()
        },
        "no_smooth_speedup": timings["unfused"] / timings["fused_no_smooth"],
        "maxpool_overhead_ratio": (
            timings["fused_maxpool"] / timings["fused_no_smooth"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, nargs="+", default=(16, 64, 256, 512))
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("benchmark_sm89_fused_preparation.py requires SM89")
    if (
        len(args.blocks) < 2
        or len(args.blocks) != len(set(args.blocks))
        or any(block <= 0 for block in args.blocks)
    ):
        raise ValueError("--blocks requires at least two unique positive lengths")
    if args.heads <= 0 or args.prefix_tokens < 0:
        raise ValueError("--heads must be positive and --prefix-tokens nonnegative")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative; iterations and repeats must be positive")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    cases: list[dict[str, object]] = []
    for query_block in (64, 128):
        for blocks in args.blocks:
            cases.append(
                _case(
                    query_block=query_block,
                    blocks=blocks,
                    heads=args.heads,
                    prefix_tokens=args.prefix_tokens,
                    dtype=dtype,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()

    fits: dict[str, dict[str, float]] = {}
    gates: dict[str, object] = {}
    for query_block in (64, 128):
        selected = [case for case in cases if case["query_block"] == query_block]
        for provider in (
            "unfused",
            "fused_no_smooth",
            "fused_smooth",
            "fused_maxpool",
        ):
            points = [
                {
                    "video_tokens": float(case["video_tokens"]),
                    "milliseconds": float(case[provider]["milliseconds"]),
                }
                for case in selected
            ]
            fits[f"q{query_block}_{provider}"] = {
                "slope_ms_per_1k_video_tokens": _slope(points)
            }
        pointwise = all(float(case["no_smooth_speedup"]) >= 1.0 for case in selected)
        slope_faster = (
            fits[f"q{query_block}_fused_no_smooth"]["slope_ms_per_1k_video_tokens"]
            <= fits[f"q{query_block}_unfused"]["slope_ms_per_1k_video_tokens"]
        )
        gates[f"q{query_block}"] = {
            "passed": pointwise and slope_faster,
            "pointwise_faster": pointwise,
            "slope_no_regression": slope_faster,
        }
    print(
        json.dumps(
            {
                "schema": "anemoi.benchmark.sm89_fused_preparation.v2",
                "device": torch.cuda.get_device_name(),
                "dtype": args.dtype,
                "cases": cases,
                "fits": fits,
                "gates": gates,
                "passed": all(gate["passed"] for gate in gates.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
