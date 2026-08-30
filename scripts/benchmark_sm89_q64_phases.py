"""Benchmark reproducible SM89 Q64 FP16, INT8, and mixed phase launches."""

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
    # One TP=4 local rank of the 56-head MiniMax-H3 720p video sequence:
    # 37 latent frames x 16 K64 blocks/frame = 592 blocks = 37,888 tokens.
    "h3_720p": {
        "heads": 14,
        "tokens": 37_888,
        "low_blocks": 95,
        "high_blocks": 24,
    },
}
_CASES = ("fp16", "int8", "int8_fp16")


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


def _make_operands(profile: dict[str, int], seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    batch = 1
    heads = profile["heads"]
    tokens = profile["tokens"]
    blocks = tokens // 64

    q16 = torch.randn((batch, heads, tokens, 128), device=device, dtype=torch.float16)
    k16 = torch.randn_like(q16)
    v16 = torch.randn_like(q16)
    q8 = torch.randint(-8, 9, q16.shape, device=device, dtype=torch.int8)
    k8 = torch.randint(-8, 9, k16.shape, device=device, dtype=torch.int8)
    # Setup is outside the timed region, but use the production permutation
    # and E4M3 conversion so the measured mainloop sees its real V layout.
    v8, v_scale = resolve_mixed_attention_operator("preprocess_v_fp8")(v16)
    q_scale = torch.full(
        (batch, heads, blocks), 1.0 / 64.0, device=device, dtype=torch.float32
    )
    k_scale = torch.full_like(q_scale, 1.0 / 64.0)

    row = (
        torch.arange(blocks, device=device, dtype=torch.int32)[None, :]
        + 13 * torch.arange(blocks, device=device, dtype=torch.int32)[:, None]
    ) % blocks
    route = row[None, None, :, :].expand(batch, heads, -1, -1).contiguous()
    valid = torch.full((batch, blocks), 64, device=device, dtype=torch.int32)
    return {
        "q16": q16,
        "k16": k16,
        "v16": v16,
        "q8": q8,
        "k8": k8,
        "v8": v8,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "route": route,
        "valid": valid,
    }


def _make_cases(
    tensors: dict[str, torch.Tensor], profile: dict[str, int]
) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    fp16_op = resolve_mixed_attention_operator("k64_fp16_attention_forward")
    int8_op = resolve_mixed_attention_operator("k64_fp8_attention_forward")
    mixed_op = resolve_mixed_attention_operator("k64_mixed_attention_forward")
    q16 = tensors["q16"]
    blocks = q16.size(2) // 64
    selected = profile["low_blocks"] + profile["high_blocks"]
    if selected > blocks:
        raise ValueError("profile selects more K64 blocks than it contains")
    shape = (q16.size(0), q16.size(1), blocks)
    full_counts = torch.full(shape, selected, device="cuda", dtype=torch.int32)
    low_counts = torch.full(
        shape, profile["low_blocks"], device="cuda", dtype=torch.int32
    )
    high_counts = torch.full(
        shape, profile["high_blocks"], device="cuda", dtype=torch.int32
    )
    scale = 1.0 / math.sqrt(128)

    def fp16() -> tuple[torch.Tensor, torch.Tensor]:
        return fp16_op(
            tensors["q16"], tensors["k16"], tensors["v16"],
            tensors["route"], full_counts, tensors["valid"], scale,
        )

    def int8() -> tuple[torch.Tensor, torch.Tensor]:
        return int8_op(
            tensors["q8"], tensors["k8"], tensors["v8"], tensors["route"],
            full_counts, tensors["q_scale"], tensors["k_scale"],
            tensors["v_scale"], tensors["valid"], scale,
        )

    def int8_fp16() -> tuple[torch.Tensor, torch.Tensor]:
        return mixed_op(
            tensors["q8"], tensors["k8"], tensors["v8"], tensors["q16"],
            tensors["k16"], tensors["v16"], tensors["route"], low_counts,
            tensors["route"], high_counts, tensors["q_scale"],
            tensors["k_scale"], tensors["v_scale"], tensors["valid"], 0, scale,
        )

    return {"fp16": fp16, "int8": int8, "int8_fp16": int8_fp16}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default="h3_720p")
    parser.add_argument("--cases", nargs="+", choices=_CASES, default=list(_CASES))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions <= 0:
        parser.error("warmups must be nonnegative and repetitions must be positive")
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
    cases = _make_cases(tensors, profile)
    results: dict[str, object] = {}
    for name in args.cases:
        timing, (output, lse) = _time_cuda(
            cases[name], args.warmups, args.repetitions
        )
        results[name] = {
            "timing": timing,
            "output_sha256": _sha256_tensor(output),
            "lse_sha256": _sha256_tensor(lse),
            "output_finite": bool(torch.isfinite(output).all().item()),
            "lse_finite": bool(torch.isfinite(lse).all().item()),
        }

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "schema": "mpa.benchmark.sm89_q64_phases.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": properties.name,
        "compute_capability": "8.9",
        "profile": args.profile,
        "shape": {"batch": 1, **profile, "head_dim": 128, "block": 64},
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
