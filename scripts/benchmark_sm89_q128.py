"""Compare SM89 Q64/Q128 pure floors and ordered INT8-to-FP16 composition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anemoi.layers.attention.mpa.build_identity import (  # noqa: E402
    import_native_extension,
    resolve_mixed_attention_operator,
)


_PROFILES = {
    "smoke": {"heads": 2, "tokens": 1024, "low_blocks": 12, "high_blocks": 4},
    # One TP=4 local rank of the 56-head MiniMax-H3 720p 5-second video.
    "h3_5s": {
        "heads": 14,
        "tokens": 37_888,
        "low_blocks": 95,
        "high_blocks": 24,
    },
    # The matching 10-second geometry has 72 latent frames instead of 37.
    "h3_10s": {
        "heads": 14,
        "tokens": 73_728,
        "low_blocks": 166,
        "high_blocks": 41,
    },
}
_PROFILES["h3_720p"] = _PROFILES["h3_5s"]
_CASES = (
    "q64_fp16",
    "q128_fp16",
    "q64_int8",
    "q128_int8",
    "q64_mixed",
    "q128_mixed",
)
_PURE_PHASES = ("fp16", "int8")


def _sha256_tensor(tensor: torch.Tensor) -> str:
    host = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(host.tobytes(order="C")).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _time_cuda(
    operation: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor]]:
    result = operation()
    for _ in range(warmups):
        result = operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return (
        {
            "min_ms": min(samples),
            "p20_ms": _percentile(samples, 0.20),
            "median_ms": statistics.median(samples),
            "p80_ms": _percentile(samples, 0.80),
            "max_ms": max(samples),
            "mean_ms": statistics.fmean(samples),
            "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        },
        result,
    )


def _time_interleaved(
    operations: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]],
    warmups: int,
    repetitions: int,
    *,
    reverse: bool,
) -> dict[str, dict[str, float]]:
    names = list(operations)
    if reverse:
        names.reverse()
    for _ in range(warmups):
        for name in names:
            operations[name]()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    for _ in range(repetitions):
        for name in names:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operations[name]()
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end)))
    return {
        name: {
            "min_ms": min(values),
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for name, values in samples.items()
    }


def _linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) < 2:
        raise ValueError("a line fit requires at least two points")
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0:
        raise ValueError("line-fit stage counts must differ")
    slope = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    ) / denominator
    return {
        "intercept_ms": mean_y - slope * mean_x,
        "slope_ms_per_stage": slope,
    }


def _evaluate_pure_phase_gate(
    fits: dict[str, dict[str, object]],
    *,
    suffix: str,
    max_q128_over_q64_slope: float,
) -> dict[str, object]:
    phases = {}
    for phase in _PURE_PHASES:
        q64_name = f"q64_{phase}{suffix}"
        q128_name = f"q128_{phase}{suffix}"
        if q64_name not in fits or q128_name not in fits:
            return {
                "evaluated": False,
                "passed": False,
                "reason": f"missing pure phase fits: {q64_name}, {q128_name}",
                "reference": "same-run Q64 pure phase",
                "max_q128_over_q64_slope": max_q128_over_q64_slope,
                "phases": {},
            }
        q64_slope = float(fits[q64_name]["slope_ms_per_stage"])
        q128_slope = float(fits[q128_name]["slope_ms_per_stage"])
        ratio = q128_slope / q64_slope if q64_slope > 0.0 else math.inf
        phases[phase] = {
            "q64_slope_ms_per_stage": q64_slope,
            "q128_slope_ms_per_stage": q128_slope,
            "q128_over_q64_slope": ratio,
            "passed": q128_slope > 0.0 and ratio <= max_q128_over_q64_slope,
        }
    return {
        "evaluated": True,
        "passed": all(bool(phase["passed"]) for phase in phases.values()),
        "reason": None,
        "reference": "same-run Q64 pure phase",
        "max_q128_over_q64_slope": max_q128_over_q64_slope,
        "phases": phases,
    }


def _evaluate_phase_transfer(
    fits: dict[str, dict[str, object]],
    *,
    pure_phase_passed: bool,
) -> dict[str, object]:
    results = {}
    for query_block in (64, 128):
        mixed = fits[f"q{query_block}_mixed"]
        int8 = fits[f"q{query_block}_int8_floor"]
        fp16 = fits[f"q{query_block}_fp16_floor"]
        expected = float(fp16["slope_ms_per_stage"]) - float(
            int8["slope_ms_per_stage"]
        )
        measured = float(mixed["slope_ms_per_stage"])
        residual = measured - expected
        repeatability_band = mixed["paired_slope_band_ms_per_stage"]
        eligible = pure_phase_passed and repeatability_band is not None
        results[f"q{query_block}"] = {
            "eligible": eligible,
            "passed": (
                residual <= float(repeatability_band) if eligible else None
            ),
            "blocked_reason": (
                None
                if eligible
                else (
                    "pure phase gate failed"
                    if not pure_phase_passed
                    else "paired provider order is required"
                )
            ),
            "measured_transfer_slope_ms_per_stage": measured,
            "standalone_int8_slope_ms_per_stage": float(
                int8["slope_ms_per_stage"]
            ),
            "standalone_fp16_slope_ms_per_stage": float(
                fp16["slope_ms_per_stage"]
            ),
            "expected_transfer_slope_ms_per_stage": expected,
            "residual_ms_per_stage": residual,
            "paired_repeatability_band_ms_per_stage": repeatability_band,
            "one_sided_no_regression_gate": True,
        }
    eligible = pure_phase_passed and all(
        bool(result["eligible"]) for result in results.values()
    )
    return {
        "eligible": eligible,
        "passed": (
            all(bool(result["passed"]) for result in results.values())
            if eligible
            else None
        ),
        "results": results,
    }


def _evaluate_single_phase_specialization_control(
    fits: dict[str, dict[str, object]],
    *,
    pure_phase_passed: bool,
) -> dict[str, object]:
    results = {}
    for query_block in (64, 128):
        for phase in _PURE_PHASES:
            pure = fits[f"q{query_block}_{phase}_floor"]
            composed = fits[f"q{query_block}_composed_{phase}_only"]
            pure_slope = float(pure["slope_ms_per_stage"])
            composed_slope = float(composed["slope_ms_per_stage"])
            residual = composed_slope - pure_slope
            pure_band = pure["paired_slope_band_ms_per_stage"]
            composed_band = composed["paired_slope_band_ms_per_stage"]
            repeatability_band = (
                float(pure_band) + float(composed_band)
                if pure_band is not None and composed_band is not None
                else None
            )
            eligible = pure_phase_passed and repeatability_band is not None
            results[f"q{query_block}_{phase}"] = {
                "eligible": eligible,
                "passed": (
                    residual <= repeatability_band if eligible else None
                ),
                "blocked_reason": (
                    None
                    if eligible
                    else (
                        "pure phase gate failed"
                        if not pure_phase_passed
                        else "paired provider order is required"
                    )
                ),
                "pure_slope_ms_per_stage": pure_slope,
                "composed_specialization_slope_ms_per_stage": composed_slope,
                "composed_specialization_over_pure_slope": (
                    composed_slope / pure_slope
                    if pure_slope > 0.0
                    else math.inf
                ),
                "residual_ms_per_stage": residual,
                "combined_repeatability_band_ms_per_stage": repeatability_band,
                "one_sided_no_regression_gate": True,
                "diagnostic_only": True,
            }
    eligible = pure_phase_passed and all(
        bool(result["eligible"]) for result in results.values()
    )
    return {
        "eligible": eligible,
        "passed": (
            all(bool(result["passed"]) for result in results.values())
            if eligible
            else None
        ),
        "results": results,
    }


def _make_operands(profile: dict[str, int], seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    batch = 1
    heads = profile["heads"]
    tokens = profile["tokens"]
    if tokens % 128:
        raise ValueError("Q64/Q128 comparison requires a multiple of 128 tokens")
    key_blocks = tokens // 64

    q16 = torch.randn((batch, heads, tokens, 128), device="cuda", dtype=torch.float16)
    k16 = torch.randn_like(q16)
    v16 = torch.randn_like(q16)
    q8 = torch.randint(-8, 9, q16.shape, device="cuda", dtype=torch.int8)
    k8 = torch.randint(-8, 9, q16.shape, device="cuda", dtype=torch.int8)
    v8, v_scale = resolve_mixed_attention_operator("preprocess_v_fp8")(v16)
    k_scale = torch.full(
        (batch, heads, key_blocks), 1.0 / 64.0,
        device="cuda", dtype=torch.float32,
    )
    valid = torch.full((batch, key_blocks), 64, device="cuda", dtype=torch.int32)
    return {
        "q16": q16,
        "k16": k16,
        "v16": v16,
        "q8": q8,
        "k8": k8,
        "v8": v8,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "valid": valid,
    }


def _make_route(
    batch: int,
    heads: int,
    query_blocks: int,
    query_block: int,
    key_blocks: int,
) -> torch.Tensor:
    query_row = torch.arange(query_blocks, device="cuda", dtype=torch.int32)
    key_col = torch.arange(key_blocks, device="cuda", dtype=torch.int32)
    # Give both Q64 rows covered by one Q128 CTA the same route. This makes
    # output deltas attributable to kernel organization instead of sparsity.
    q128_row = query_row * query_block // 128
    route = (key_col[None, :] + 13 * q128_row[:, None]) % key_blocks
    return route[None, None, :, :].expand(batch, heads, -1, -1).contiguous()


def _make_cases(
    tensors: dict[str, torch.Tensor], profile: dict[str, int]
) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    q64_fp16_op = resolve_mixed_attention_operator("k64_fp16_attention_forward")
    q128_fp16_op = resolve_mixed_attention_operator(
        "q128_k64_fp16_attention_forward"
    )
    q64_int8_op = resolve_mixed_attention_operator("k64_fp8_attention_forward")
    q128_int8_op = resolve_mixed_attention_operator(
        "q128_k64_fp8_attention_forward"
    )
    q64_mixed_op = resolve_mixed_attention_operator("k64_mixed_attention_forward")
    q128_mixed_op = resolve_mixed_attention_operator(
        "q128_k64_mixed_attention_forward"
    )
    batch, heads, tokens, _ = tensors["q8"].shape
    key_blocks = tokens // 64
    selected = profile["low_blocks"] + profile["high_blocks"]
    if selected > key_blocks:
        raise ValueError("profile selects more K64 blocks than it contains")
    scale = 1.0 / math.sqrt(128)
    cases: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]] = {}

    for query_block, fp16_op, int8_op, mixed_op in (
        (64, q64_fp16_op, q64_int8_op, q64_mixed_op),
        (128, q128_fp16_op, q128_int8_op, q128_mixed_op),
    ):
        query_blocks = tokens // query_block
        route = _make_route(
            batch, heads, query_blocks, query_block, key_blocks
        )
        shape = (batch, heads, query_blocks)
        q_scale = torch.full(shape, 1.0 / 64.0, device="cuda", dtype=torch.float32)
        full_counts = torch.full(shape, selected, device="cuda", dtype=torch.int32)
        low_counts = torch.full(
            shape, profile["low_blocks"], device="cuda", dtype=torch.int32
        )
        high_counts = torch.full(
            shape, profile["high_blocks"], device="cuda", dtype=torch.int32
        )

        def fp16(
            op: Callable = fp16_op,
            route_: torch.Tensor = route,
            counts: torch.Tensor = full_counts,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return op(
                tensors["q16"], tensors["k16"], tensors["v16"],
                route_, counts, tensors["valid"], scale,
            )

        def int8(
            op: Callable = int8_op,
            route_: torch.Tensor = route,
            counts: torch.Tensor = full_counts,
            q_scale_: torch.Tensor = q_scale,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return op(
                tensors["q8"], tensors["k8"], tensors["v8"], route_, counts,
                q_scale_, tensors["k_scale"], tensors["v_scale"],
                tensors["valid"], scale,
            )

        def mixed(
            op: Callable = mixed_op,
            route_: torch.Tensor = route,
            low_counts_: torch.Tensor = low_counts,
            high_counts_: torch.Tensor = high_counts,
            q_scale_: torch.Tensor = q_scale,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return op(
                tensors["q8"], tensors["k8"], tensors["v8"],
                tensors["q16"], tensors["k16"], tensors["v16"],
                route_, low_counts_, route_, high_counts_, q_scale_,
                tensors["k_scale"], tensors["v_scale"], tensors["valid"],
                0, scale,
            )

        cases[f"q{query_block}_fp16"] = fp16
        cases[f"q{query_block}_int8"] = int8
        cases[f"q{query_block}_mixed"] = mixed
    return cases


def _make_sweep_cases(
    tensors: dict[str, torch.Tensor],
    profile: dict[str, int],
    stages: tuple[int, ...],
    mode: str,
    selected_cases: tuple[str, ...],
) -> tuple[
    dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]],
    dict[str, list[tuple[float, str]]],
]:
    operations = {}
    series: dict[str, list[tuple[float, str]]] = {}
    if mode == "total":
        for stage in stages:
            point_profile = {**profile, "low_blocks": stage, "high_blocks": 0}
            for case, operation in _make_cases(tensors, point_profile).items():
                if case not in selected_cases:
                    continue
                name = f"{case}_total_{stage}"
                operations[name] = operation
                series.setdefault(case, []).append((float(stage), name))
        return operations, series

    if mode == "phase":
        for stage in stages:
            if stage == 0:
                raise ValueError("phase sweep stages must be positive")
            int8_profile = {**profile, "low_blocks": stage, "high_blocks": 0}
            fp16_profile = {**profile, "low_blocks": 0, "high_blocks": stage}
            int8_cases = _make_cases(tensors, int8_profile)
            fp16_cases = _make_cases(tensors, fp16_profile)
            for query_block in (64, 128):
                names_and_operations = (
                    (f"q{query_block}_int8_floor", int8_cases[f"q{query_block}_int8"]),
                    (f"q{query_block}_fp16_floor", fp16_cases[f"q{query_block}_fp16"]),
                    (
                        f"q{query_block}_composed_int8_only",
                        int8_cases[f"q{query_block}_mixed"],
                    ),
                    (
                        f"q{query_block}_composed_fp16_only",
                        fp16_cases[f"q{query_block}_mixed"],
                    ),
                )
                for case, operation in names_and_operations:
                    name = f"{case}_{stage}"
                    operations[name] = operation
                    series.setdefault(case, []).append((float(stage), name))
        return operations, series

    total = profile["low_blocks"] + profile["high_blocks"]
    for stage in stages:
        if stage > total:
            raise ValueError("transfer stage exceeds the profile's selected blocks")
        transfer_profile = {
            **profile,
            "low_blocks": total - stage,
            "high_blocks": stage,
        }
        transfer_cases = _make_cases(tensors, transfer_profile)
        for query_block in (64, 128):
            case = f"q{query_block}_mixed"
            name = f"{case}_transfer_{stage}"
            operations[name] = transfer_cases[case]
            series.setdefault(case, []).append((float(stage), name))
        if stage == 0:
            continue
        floor_profile = {**profile, "low_blocks": stage, "high_blocks": 0}
        floor_cases = _make_cases(tensors, floor_profile)
        for query_block in (64, 128):
            for phase in ("int8", "fp16"):
                case = f"q{query_block}_{phase}_floor"
                name = f"{case}_{stage}"
                operations[name] = floor_cases[f"q{query_block}_{phase}"]
                series.setdefault(case, []).append((float(stage), name))
    return operations, series


def _is_pure_series(name: str) -> bool:
    return name.endswith("_floor") or name in {
        "q64_fp16",
        "q128_fp16",
        "q64_int8",
        "q128_int8",
    }


def _summarize_timing_orders(
    operations: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]],
    timing_orders: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "median_ms": statistics.fmean(
                order[name]["median_ms"] for order in timing_orders.values()
            ),
            "min_ms": min(order[name]["min_ms"] for order in timing_orders.values()),
        }
        for name in operations
    }


def _fit_sweep_series(
    series: dict[str, list[tuple[float, str]]],
    timing: dict[str, dict[str, float]],
    timing_orders: dict[str, dict[str, dict[str, float]]],
    provider_order: str,
) -> dict[str, dict[str, object]]:
    fits = {}
    for name, points in series.items():
        if any(operation not in timing for _, operation in points):
            continue
        fit_points = [
            (stage, timing[operation]["median_ms"])
            for stage, operation in points
        ]
        selected = _linear_fit(fit_points)
        fits_by_order = {
            order: {
                **_linear_fit(
                    [
                        (stage, values[operation]["median_ms"])
                        for stage, operation in points
                    ]
                ),
                "points": [
                    [stage, values[operation]["median_ms"]]
                    for stage, operation in points
                ],
            }
            for order, values in timing_orders.items()
        }
        selected["points"] = [[x, y] for x, y in fit_points]
        selected["fits_by_order"] = fits_by_order
        selected["paired_slope_band_ms_per_stage"] = (
            abs(
                fits_by_order["forward"]["slope_ms_per_stage"]
                - fits_by_order["reverse"]["slope_ms_per_stage"]
            )
            if provider_order == "paired"
            else None
        )
        fits[name] = selected
    return fits


def _run_sweep(
    tensors: dict[str, torch.Tensor],
    profile: dict[str, int],
    stages: tuple[int, ...],
    mode: str,
    warmups: int,
    repetitions: int,
    provider_order: str,
    selected_cases: tuple[str, ...],
    max_pure_slope_ratio: float,
) -> dict[str, object]:
    operations, series = _make_sweep_cases(
        tensors, profile, stages, mode, selected_cases
    )
    orders = (
        (("forward", False), ("reverse", True))
        if provider_order == "paired"
        else ((provider_order, provider_order == "reverse"),)
    )
    pure_series = {name: points for name, points in series.items() if _is_pure_series(name)}
    pure_operation_names = {
        operation for points in pure_series.values() for _, operation in points
    }
    pure_operations = {
        name: operation
        for name, operation in operations.items()
        if name in pure_operation_names
    }
    composition_operations = {
        name: operation
        for name, operation in operations.items()
        if name not in pure_operation_names
    }

    pure_timing_orders = {
        name: _time_interleaved(
            pure_operations, warmups, repetitions, reverse=reverse
        )
        for name, reverse in orders
    }
    pure_timing = _summarize_timing_orders(
        pure_operations, pure_timing_orders
    )
    pure_fits = _fit_sweep_series(
        pure_series, pure_timing, pure_timing_orders, provider_order
    )
    pure_suffix = "_floor" if mode in ("phase", "transfer") else ""
    pure_phase_gate = _evaluate_pure_phase_gate(
        pure_fits,
        suffix=pure_suffix,
        max_q128_over_q64_slope=max_pure_slope_ratio,
    )

    composition_skipped = bool(composition_operations) and not bool(
        pure_phase_gate["passed"]
    )
    composition_timing_orders = {
        name: _time_interleaved(
            composition_operations, warmups, repetitions, reverse=reverse
        )
        for name, reverse in orders
    } if composition_operations and not composition_skipped else {
        name: {} for name, _ in orders
    }
    timing_orders = {
        name: {**pure_timing_orders[name], **composition_timing_orders[name]}
        for name, _ in orders
    }
    timing = {
        **pure_timing,
        **(
            _summarize_timing_orders(
                composition_operations, composition_timing_orders
            )
            if not composition_skipped
            else {}
        ),
    }
    fits = _fit_sweep_series(series, timing, timing_orders, provider_order)
    phase_transfer: dict[str, object] = {}
    specialization_control: dict[str, object] = {}
    if mode == "phase" and not composition_skipped:
        specialization_control = _evaluate_single_phase_specialization_control(
            fits, pure_phase_passed=bool(pure_phase_gate["passed"])
        )
    elif mode == "phase":
        specialization_control = {
            "eligible": False,
            "passed": None,
            "blocked_reason": "pure phase gate failed",
            "results": {},
            "diagnostic_only": True,
        }
    if mode == "transfer" and not composition_skipped:
        phase_transfer = _evaluate_phase_transfer(
            fits, pure_phase_passed=bool(pure_phase_gate["passed"])
        )
    elif mode == "transfer":
        phase_transfer = {
            "eligible": False,
            "passed": None,
            "blocked_reason": "pure phase gate failed",
            "results": {},
        }
    return {
        "mode": mode,
        "stages": stages,
        "provider_order": provider_order,
        "execution_order": ("pure", "composition"),
        "composition_skipped": composition_skipped,
        "timing": timing,
        "fits": fits,
        "pure_phase_gate": pure_phase_gate,
        "single_phase_specialization_control": specialization_control,
        "phase_transfer": phase_transfer,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default="h3_720p")
    parser.add_argument("--cases", nargs="+", choices=_CASES, default=list(_CASES))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--sweep-mode", choices=("total", "phase", "transfer"))
    parser.add_argument(
        "--sweep-stages",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        default=(4, 8, 16, 32, 64, 96),
    )
    parser.add_argument(
        "--provider-order", choices=("forward", "reverse", "paired"),
        default="paired",
    )
    parser.add_argument(
        "--max-pure-slope-ratio",
        type=float,
        default=1.07,
        help=(
            "maximum accepted Q128/Q64 stage-slope ratio for every pure phase; "
            "the mixed transfer gate is blocked until this passes"
        ),
    )
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="return a nonzero status when an evaluated performance gate fails",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions <= 0:
        parser.error("warmups must be nonnegative and repetitions must be positive")
    if (
        not args.sweep_stages
        or any(stage < 0 for stage in args.sweep_stages)
        or len(args.sweep_stages) != len(set(args.sweep_stages))
    ):
        parser.error("sweep stages must be unique nonnegative integers")
    if args.sweep_mode == "total" and any(stage == 0 for stage in args.sweep_stages):
        parser.error("total sweep stages must be positive")
    if args.sweep_mode == "phase" and any(stage == 0 for stage in args.sweep_stages):
        parser.error("phase sweep stages must be positive")
    if not math.isfinite(args.max_pure_slope_ratio) or args.max_pure_slope_ratio <= 0:
        parser.error("max pure slope ratio must be a finite positive number")
    return args


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this benchmark requires an SM89 GPU")
    extension = import_native_extension("attention")
    profile = dict(_PROFILES[args.profile])
    tensors = _make_operands(profile, args.seed)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    common = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": properties.name,
        "compute_capability": "8.9",
        "profile": args.profile,
        "shape": {"batch": 1, **profile, "head_dim": 128, "key_block": 64},
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "seed": args.seed,
    }
    if args.sweep_mode is not None:
        payload = {
            "schema": "mpa.benchmark.sm89_q128_phase_sweep.v2",
            **common,
            "sweep": _run_sweep(
                tensors, profile, args.sweep_stages, args.sweep_mode,
                args.warmups, args.repetitions, args.provider_order,
                tuple(args.cases), args.max_pure_slope_ratio,
            ),
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        print(rendered)
        if args.enforce_gates:
            pure_gate = payload["sweep"]["pure_phase_gate"]
            phase_gate = payload["sweep"]["single_phase_specialization_control"]
            transfer_gate = payload["sweep"]["phase_transfer"]
            if not pure_gate["evaluated"] or not pure_gate["passed"]:
                return 2
            if phase_gate and phase_gate["eligible"] and not phase_gate["passed"]:
                return 3
            if transfer_gate and transfer_gate["eligible"] and not transfer_gate["passed"]:
                return 4
        return 0
    cases = _make_cases(tensors, profile)
    results: dict[str, object] = {}
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in args.cases:
        timing, output = _time_cuda(cases[name], args.warmups, args.repetitions)
        outputs[name] = output
        results[name] = {
            "timing": timing,
            "output_sha256": _sha256_tensor(output[0]),
            "lse_sha256": _sha256_tensor(output[1]),
            "output_finite": bool(torch.isfinite(output[0]).all().item()),
            "lse_finite": bool(torch.isfinite(output[1]).all().item()),
        }
    comparisons = {}
    for precision in ("fp16", "int8", "mixed"):
        q64_name = f"q64_{precision}"
        q128_name = f"q128_{precision}"
        if q64_name in outputs and q128_name in outputs:
            q64_output, q64_lse = outputs[q64_name]
            q128_output, q128_lse = outputs[q128_name]
            comparisons[precision] = {
                "output_max_abs_diff": float(
                    (q64_output.float() - q128_output.float()).abs().max().item()
                ),
                "lse_max_abs_diff": float((q64_lse - q128_lse).abs().max().item()),
                "q128_over_q64_median": (
                    results[q128_name]["timing"]["median_ms"]
                    / results[q64_name]["timing"]["median_ms"]
                ),
            }

    payload = {
        "schema": "mpa.benchmark.sm89_q128.v1",
        **common,
        "results": results,
        "comparisons": comparisons,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
