#!/usr/bin/env python3
"""Run one MiniMax-H3 compressed-weight quality candidate on 4x RTX 4090.

The runner combines SolEngine's pruned-FP8 DiT loader with its lossless
Ulysses packed-QKV exchange.  Every candidate uses the same checkpoint,
conditioning, model fusions, sequence sharding and decode path; only the
post-Ulysses attention callable changes. MPA assigns retained video blocks
between the FP8 and FP16 phases at an 80/20 ratio by default.

The bundled SolEngine runtime snapshot is imported read-only. Persistent
artifacts go to the requested output directory; compilation and runtime caches
default to ``/dev/shm``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOL_ROOT = ROOT / "integrations/minimax_h3/solengine"
DEFAULT_DIFFUSERS = Path(
    os.environ.get("H3_DIFFUSERS_CHECKOUT", ROOT / "external/diffusers")
)
DEFAULT_MODEL_ROOT = Path(
    os.environ.get("H3_MODEL_ROOT", ROOT / "external/MiniMax-H3-diffusers")
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "H3_DIT_CHECKPOINT",
        ROOT / "external/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    )
)
DEFAULT_CONDITIONING = (
    ROOT / "assets/conditioning/official-example-1.pt"
)
CHECKPOINT_BYTES = 20_958_205_608
CHECKPOINT_SHA256 = (
    "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993"
)
PROMPT_SHA256 = (
    "98f36b879692095e099ae824c18d9e93e7006a490e082fd474a5f531769dcf06"
)
PROMPTS = {
    "official-example-1": (DEFAULT_CONDITIONING, PROMPT_SHA256),
}


SM89_MAINLINE_CANDIDATE = "mpa-sm89-regular2d-mixed"
_MPA_PROFILES: dict[str, dict[str, Any]] = {
    SM89_MAINLINE_CANDIDATE: {
        "adaptive_tile": True,
        "tile_shape": (8, 8),
        "video_sparsity_ratio": 0.90,
        "layer_sparsity_bands": ((18, 34, 0.85), (34, 50, 0.65)),
        "average_sparse_layer_sparsity_ratio": 0.80,
        "precision": {"fp8": 0.8, "fp16": 0.2},
        "route_score": (
            "pooled-QK row-softmax probability; exact per-head global top-k; "
            "same-frame legal 2-D cross anchors consume the same budget"
        ),
        "tile_note": (
            "regular request-level selector: attention grid 24x42 -> 8x7; "
            "24x40 -> 8x8; both execute through physical Q64xK64"
        ),
        "skip_compensation": "disabled; unselected blocks are dropped",
    }
}
CANDIDATES = ("dense", "official-sol", SM89_MAINLINE_CANDIDATE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=CANDIDATES,
        default=SM89_MAINLINE_CANDIDATE,
        help=(
            "attention candidate; defaults to the current SM89 mainline: "
            "regular adaptive 2-D pooling, probability-global routing, legal "
            "2-D cross anchors, rising layer budget, FP8/FP16=80/20, and no "
            "skip compensation"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sol-root", type=Path, default=DEFAULT_SOL_ROOT)
    parser.add_argument("--diffusers-src", type=Path, default=DEFAULT_DIFFUSERS)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--conditioning", type=Path)
    parser.add_argument(
        "--prompt-id", choices=tuple(PROMPTS), default="official-example-1"
    )
    parser.add_argument(
        "--fp8-ratio",
        type=float,
        help="retained sparse blocks assigned to the SM89 FP8 phase",
    )
    parser.add_argument(
        "--fp16-ratio",
        type=float,
        help="retained sparse blocks assigned to the exact FP16 rescue phase",
    )
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="50 for the quality workload; 2 is the supported load/memory smoke",
    )
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="decode an already completed denoised_state.pt without torchrun",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved candidate without initializing CUDA/distributed",
    )
    return parser


def _resolved_candidate(
    name: str,
    fp8_ratio: float | None = None,
    fp16_ratio: float | None = None,
) -> dict[str, Any]:
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}")
    if (fp8_ratio is None) != (fp16_ratio is None):
        raise ValueError("--fp8-ratio and --fp16-ratio must be provided together")
    if fp8_ratio is not None:
        assert fp16_ratio is not None
        if (
            not math.isfinite(fp8_ratio)
            or not math.isfinite(fp16_ratio)
            or min(fp8_ratio, fp16_ratio) <= 0.0
            or fp16_ratio <= 0.0
            or not math.isclose(fp8_ratio + fp16_ratio, 1.0, abs_tol=1e-6)
        ):
            raise ValueError(
                "FP8/FP16 ratios must be finite, positive, and sum to one"
            )
    if name == "dense":
        if fp8_ratio is not None:
            raise ValueError("precision overrides are only valid for MPA")
        return {"kind": "dense"}
    if name == "official-sol":
        if fp8_ratio is not None:
            raise ValueError("official Sol is pinned to its published precision policy")
        return {
            "kind": "official-sol",
            "tau": 1.0,
            "threshold_type": "diag",
            "dense_first_steps": 10,
            "dense_first_layers": 2,
            "sink_mode": "prefix",
        }
    profile = dict(_MPA_PROFILES[name])
    if fp8_ratio is not None:
        assert fp16_ratio is not None
        profile["precision"] = {
            "fp8": float(fp8_ratio),
            "fp16": float(fp16_ratio),
        }
    return {
        "kind": "mpa",
        "precision": profile.pop(
            "precision", {"fp8": 0.8, "fp16": 0.2}
        ),
        "dense_first_steps": 10,
        "dense_first_layers": 2,
        "scheduled_dense": "original-dtype torch SDPA",
        "prefix_query_overwrite": "original-dtype torch SDPA",
        **profile,
    }


def _prepend(path: Path) -> None:
    value = str(path.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_tree_sha256(root: Path) -> str:
    """Hash Python source paths and bytes for reproducible provider provenance."""

    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"Python source tree is empty: {root}")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor) -> str:
    import torch

    value = tensor.detach().cpu().contiguous().view(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _state_value(state, name: str):
    value = getattr(state, name, None)
    if value is None and hasattr(state, "get"):
        value = state.get(name)
    return value


def _configure_environment(args: argparse.Namespace) -> tuple[Path, Path]:
    gb10 = args.sol_root.resolve() / "models/minimax_h3/gb10_fp8"
    optimized = args.sol_root.resolve() / "models/minimax_h3/optimized"
    required = (
        gb10 / "build.py",
        optimized / "ulysses_custom.py",
        args.diffusers_src.resolve() / "src/diffusers/__init__.py",
        args.model_root.resolve() / "transformer/config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required local sources are missing: {missing}")

    os.environ.setdefault("H3_BENCH_ROOT", str(ROOT))
    os.environ.setdefault("H3_SOL_ROOT", str(args.sol_root.resolve()))
    os.environ.setdefault("H3_SOL_GB10", str(gb10))
    os.environ.setdefault("H3_DIFFUSERS_SRC", str(args.diffusers_src.resolve() / "src"))
    os.environ.setdefault("H3_MODEL_ROOT", str(args.model_root.resolve()))
    os.environ.setdefault("H3_DIT_CHECKPOINT", str(args.checkpoint.resolve()))
    os.environ.setdefault("H3_SOL_ATTN_ROOT", str(args.sol_root.resolve() / "techniques/sparse_backends"))
    local_rank = os.environ.get("LOCAL_RANK", "0")
    os.environ.setdefault("HF_HOME", "/dev/shm/mpa-h3-fp8-hf")
    os.environ.setdefault(
        "TRITON_CACHE_DIR", f"/dev/shm/mpa-h3-fp8-triton-rank{local_rank}"
    )
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        f"/dev/shm/mpa-h3-fp8-inductor-rank{local_rank}",
    )
    os.environ.setdefault("XDG_CACHE_HOME", "/dev/shm/mpa-h3-fp8-xdg")
    os.environ.setdefault("TMPDIR", "/dev/shm")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    _prepend(ROOT / "python")
    _prepend(ROOT)
    _prepend(args.diffusers_src.resolve() / "src")
    _prepend(args.sol_root.resolve() / "techniques/sparse_backends")
    _prepend(gb10)
    return gb10, optimized


def _import_benchmark_support():
    _prepend(ROOT / "benchmarks")
    return importlib.import_module("minimax_h3_runtime")


def _warm_sparse_provider(
    candidate: dict[str, Any],
    local_heads: int,
    *,
    prefix_tokens: int,
    sequence_tokens: int,
    video_shape: tuple[int, int, int],
) -> None:
    """Compile the candidate's full-sequence operator before loading the DiT."""

    if candidate["kind"] == "dense":
        return
    import torch

    generator = torch.Generator(device="cuda").manual_seed(20260811)
    # Keep the synthetic compiler warm-up on the canonical flat-QKV fallback
    # path.  A split [S,H,3D] allocation is reserved for the live H3 handoff:
    # it is recognized as that private packed layout and must be accompanied by
    # the live transformer's scoped layout token.  Standalone dense tensors
    # exercise the same full-sequence kernels without impersonating that owner.
    q, k, v = (
        torch.randn(
            sequence_tokens,
            local_heads,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )
    if candidate["kind"] == "official-sol":
        from sol_attn import sol_attn

        output = sol_attn(
            q.unsqueeze(0).contiguous(),
            k.unsqueeze(0).contiguous(),
            v.unsqueeze(0).contiguous(),
            tau=1.0,
            thresh_type="diag",
            sink_start=0,
            sink_tokens=prefix_tokens,
        )
    else:
        attention = _make_mpa_attention(
            candidate,
            prefix_tokens=prefix_tokens,
            video_shape=video_shape,
        )
        output = attention.mpa(q, k, v, layer=2)
    torch.cuda.synchronize()
    del output, q, k, v
    gc.collect()
    torch.cuda.empty_cache()


def _make_mpa_attention(
    candidate: dict[str, Any],
    *,
    prefix_tokens: int,
    video_shape: tuple[int, int, int],
):
    from integrations.minimax_h3 import H3MPAAttention, H3MPAConfig

    precision = candidate.get(
        "precision", {"fp8": 0.8, "fp16": 0.2}
    )
    return H3MPAAttention(
        H3MPAConfig(
            video_shape=video_shape,
            prefix_tokens=prefix_tokens,
            sparsity_ratio=float(candidate.get("video_sparsity_ratio", 0.8)),
            fp8_ratio=float(precision["fp8"]),
            fp16_ratio=float(precision["fp16"]),
            dense_first_steps=int(candidate["dense_first_steps"]),
            dense_first_layers=int(candidate["dense_first_layers"]),
            layers_per_step=50,
            layer_sparsity_bands=tuple(
                tuple(band) for band in candidate.get("layer_sparsity_bands", ())
            ),
            adaptive_tile=bool(candidate.get("adaptive_tile", True)),
            tile_shape=tuple(candidate.get("tile_shape", (8, 8))),
            strict=True,
        )
    )


def _build_transformer(bench, checkpoint: Path, model_root: Path):
    import torch

    residency = bench._install_streamed_weight_only_fp8()
    from build import build_pruned_fp8_transformer

    config = json.loads((model_root / "transformer/config.json").read_text())
    started = time.perf_counter()
    transformer, info = build_pruned_fp8_transformer(
        str(checkpoint),
        config,
        device="cuda",
        fuse_qkv=False,
        quantizer="triton",
        fuse_adaln=True,
        fuse_rope=True,
        fuse_swiglu=True,
    )
    torch.cuda.synchronize()
    return transformer, {
        "wall_s_excluded": round(time.perf_counter() - started, 3),
        "resident_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "loader": info,
        "weight_only_residency": residency,
    }


def _switch_to_optimized_modules(optimized: Path) -> None:
    """Avoid GB10/optimized modules with identical unqualified names colliding."""

    for name in ("adaln", "build", "cache_line", "fusion_install", "fusions", "relayout"):
        sys.modules.pop(name, None)
    _prepend(optimized)


def _install_post_ulysses(
    transformer,
    candidate: dict[str, Any],
    *,
    prefix_tokens: int,
    video_shape: tuple[int, int, int],
):
    import sol_attn_h3
    import ulysses_custom

    attention = None
    cleanup = []
    if candidate["kind"] == "official-sol":
        os.environ["SOL_ATTN_CORRECTNESS_GATE"] = "0"
        os.environ["SOL_ATTN_STRICT"] = "1"
        attention = sol_attn_h3.install(
            transformer,
            tau=1.0,
            thresh_type="diag",
            dense_steps=10,
            dense_layers=2,
            sink_mode="prefix",
        )
        cleanup.append(lambda: sol_attn_h3.uninstall(transformer))
    elif candidate["kind"] == "mpa":
        from integrations.minimax_h3 import install_layout_observer

        attention = _make_mpa_attention(
            candidate,
            prefix_tokens=prefix_tokens,
            video_shape=video_shape,
        )
        handle = install_layout_observer(transformer, attention)
        cleanup.append(handle.remove)

    cleanup.append(
        ulysses_custom.install(
            transformer,
            packed=True,
            use_fusion=True,
            attention_fn=attention,
        )
    )
    return attention, cleanup


def _artifact_from_state(
    state,
    args: argparse.Namespace,
    *,
    prompt_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact_schema": 1,
        "model_path": str(args.model_root.resolve()),
        "prompt_sha256": prompt_sha256,
        "seed": 0,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "latents": _state_value(state, "latents").detach().float().cpu(),
        "audio_latents": _state_value(state, "audio_latents").detach().float().cpu(),
        **{
            name: _state_value(state, name)
            for name in (
                "num_condition_video_rows",
                "num_condition_audio_rows",
                "num_latent_frames",
                "latent_height",
                "latent_width",
                "num_audio_latents",
            )
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


def _decode_existing(args: argparse.Namespace) -> int:
    import torch

    output_dir = args.output_dir.resolve()
    result_path = output_dir / "benchmark.json"
    artifact_path = output_dir / "denoised_state.pt"
    if not result_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(
            "--decode-only requires benchmark.json and denoised_state.pt in output-dir"
        )
    results = json.loads(result_path.read_text())
    if results.get("candidate") != args.candidate:
        raise ValueError(
            f"saved candidate {results.get('candidate')!r} != {args.candidate!r}"
        )
    if results.get("status") not in ("denoised", "complete"):
        raise ValueError(f"saved benchmark status is {results.get('status')!r}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    bench = _import_benchmark_support()
    bench._decode_and_export(
        {args.candidate: artifact},
        results,
        output_dir,
        args.candidate,
    )
    decoded = Path(results["candidates"][args.candidate]["decode"]["output"])
    shutil.copy2(decoded, output_dir / "out.mp4")
    results["status"] = "complete"
    _write_json(result_path, results)
    return 0


def _main_distributed(args: argparse.Namespace, candidate: dict[str, Any]) -> int:
    import torch
    import torch.distributed as dist
    from diffusers.models._modeling_parallel import ContextParallelConfig

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 4:
        raise RuntimeError(f"this 4090 compressed runner requires torchrun world_size=4, got {world}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "artifacts").mkdir(exist_ok=True)
    dist.barrier()

    checkpoint_status = None
    source_provenance = None
    if rank == 0:
        if not args.checkpoint.is_file() or args.checkpoint.stat().st_size != CHECKPOINT_BYTES:
            raise RuntimeError(f"compressed checkpoint is incomplete: {args.checkpoint}")
        observed = _sha256(args.checkpoint)
        if observed != CHECKPOINT_SHA256:
            raise RuntimeError(f"compressed checkpoint SHA256 {observed} != {CHECKPOINT_SHA256}")
        checkpoint_status = observed
        source_provenance = {
            "sol_h3_integration_sha256": _sha256(
                args.sol_root.resolve()
                / "models/minimax_h3/optimized/sol_attn_h3.py"
            ),
            "sol_attn_python_tree_sha256": _python_tree_sha256(
                args.sol_root.resolve()
                / "techniques/sparse_backends/sol_attn"
            ),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        }
    values = [checkpoint_status]
    dist.broadcast_object_list(values, src=0)

    conditioning = torch.load(args.conditioning, map_location="cpu", weights_only=True)
    expected_prompt_sha256 = PROMPTS[args.prompt_id][1]
    prompt_sha256 = conditioning.get("prompt_sha256")
    if prompt_sha256 != expected_prompt_sha256:
        raise RuntimeError(
            "conditioning cache does not match --prompt-id: "
            f"{prompt_sha256!r} != {expected_prompt_sha256!r}"
        )
    prompt_embeds = conditioning["prompt_embeds"]
    text_token_tags = conditioning["text_token_tags"]
    text_tokens = int(prompt_embeds.shape[1])
    from diffusers.modular_pipelines.minimax_h3.packing import (
        MINIMAX_H3_AUDIO_CHANNELS,
        audio_latent_num_frames,
        video_latent_num_frames,
    )

    audio_tokens = int(audio_latent_num_frames(args.num_frames)) * int(
        MINIMAX_H3_AUDIO_CHANNELS
    )
    prefix_tokens = text_tokens + audio_tokens
    video_shape = (
        int(video_latent_num_frames(args.num_frames)),
        args.height // 32,
        args.width // 32,
    )
    sequence_tokens = prefix_tokens + math.prod(video_shape)

    bench = _import_benchmark_support()
    _warm_sparse_provider(
        candidate,
        local_heads=14,
        prefix_tokens=prefix_tokens,
        sequence_tokens=sequence_tokens,
        video_shape=video_shape,
    )
    transformer, build = _build_transformer(bench, args.checkpoint, args.model_root)
    denoise_pipe = bench._build_denoise_pipe(transformer)

    _switch_to_optimized_modules(args.sol_root.resolve() / "models/minimax_h3/optimized")
    from cp_plan import MINIMAX_H3_CP_PLAN, assert_no_attention_mask

    assert_no_attention_mask(transformer)
    transformer.enable_parallelism(
        config=ContextParallelConfig(ulysses_degree=4, ulysses_anything=True),
        cp_plan=MINIMAX_H3_CP_PLAN,
    )
    attention, cleanup = _install_post_ulysses(
        transformer,
        candidate,
        prefix_tokens=prefix_tokens,
        video_shape=video_shape,
    )

    run_args = argparse.Namespace(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        steps=args.steps,
        seed=0,
    )
    warm_inputs = bench._run_inputs(prompt_embeds, text_token_tags, run_args)
    warm_inputs["num_inference_steps"] = 2
    warm_started = time.perf_counter()
    warm_state = denoise_pipe(**warm_inputs)
    torch.cuda.synchronize()
    warm_wall = time.perf_counter() - warm_started
    del warm_state, warm_inputs
    gc.collect()
    torch.cuda.empty_cache()

    # A two-point scheduler request performs only one DiT evaluation, so the
    # timestep-direction observers cannot infer a subsequent request boundary
    # from a reversal.  Reinstalling the callable gives the measured request a
    # fresh step-0 clock while retaining all compiled kernels and model fusions.
    if attention is not None:
        for undo in reversed(cleanup):
            undo()
        del attention, cleanup
        attention, cleanup = _install_post_ulysses(
            transformer,
            candidate,
            prefix_tokens=prefix_tokens,
            video_shape=video_shape,
        )

    timer = bench.EvalTimer(transformer)
    timer.install()
    torch.cuda.reset_peak_memory_stats()
    run_inputs = bench._run_inputs(prompt_embeds, text_token_tags, run_args)
    dist.barrier()
    started = time.perf_counter()
    state = denoise_pipe(**run_inputs)
    torch.cuda.synchronize()
    dist.barrier()
    wall_s = time.perf_counter() - started
    timer.remove()
    samples = timer.samples_ms()
    artifact = _artifact_from_state(
        state, args, prompt_sha256=prompt_sha256
    )
    if attention is not None:
        if candidate["kind"] == "official-sol":
            attention._close_request()
        else:
            attention.close_request()
    local = {
        "rank": rank,
        "warmup_wall_s_excluded": round(warm_wall, 3),
        "resident_denoise_wall_s": round(wall_s, 3),
        "transformer_gpu_s": round(sum(samples) / 1000.0, 3),
        "per_eval_gpu_ms": [round(value, 3) for value in samples],
        "dit_evals": len(samples),
        "per_eval_gpu_ms_p50": round(statistics.median(samples), 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "video_latent_sha256": _tensor_sha256(artifact["latents"]),
        "audio_latent_sha256": _tensor_sha256(artifact["audio_latents"]),
        "attention_stats": attention.stats() if attention is not None else None,
    }
    per_rank: list[Any] = [None] * world
    dist.all_gather_object(per_rank, local)

    results = None
    if rank == 0:
        assert source_provenance is not None
        latent_path = output_dir / "denoised_state.pt"
        torch.save(artifact, latent_path)
        consistent = len({item["video_latent_sha256"] for item in per_rank}) == 1 and len(
            {item["audio_latent_sha256"] for item in per_rank}
        ) == 1
        if not consistent:
            raise RuntimeError("Ulysses ranks produced different gathered latent states")
        results = {
            "schema": "mpa.benchmark.minimax_h3_fp8_ulysses_sm89.v1",
            "status": "denoised",
            "candidate": args.candidate,
            "candidate_policy": candidate,
            "workload": {
                "resolution": [args.width, args.height],
                "generated_frames": args.num_frames,
                "delivered_frames": args.num_frames - 4,
                "fps": 24,
                "steps": args.steps,
                "dit_evals": args.steps - 1,
                "seed": 0,
                "prompt_id": args.prompt_id,
                "prompt_sha256": prompt_sha256,
                "text_tokens": text_tokens,
                "prefix_tokens": prefix_tokens,
                "sequence_tokens": sequence_tokens,
                "video_shape": list(video_shape),
            },
            "fairness": {
                "same_pruned_fp8_dit": True,
                "same_conditioning": True,
                "same_ulysses_degree": 4,
                "same_lossless_stack": [
                    "streamed_weight_only_fp8",
                    "triton_fp8_activation_quantizer",
                    "fused_adaln",
                    "fused_rope",
                    "fused_swiglu",
                    "packed_qkv_ulysses",
                ],
                "attention_precision_note": (
                    "Scheduled dense calls and sparse-call prefix-query overwrites use "
                    "the original-dtype torch SDPA, matching Sol-H3. Active native K64 "
                    "sparse attention assigns retained video blocks FP8/FP16=80/20; "
                    "exact prefix K/V is FP16. Model weights are common pruned FP8."
                ),
            },
            "provenance": {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_bytes": CHECKPOINT_BYTES,
                "checkpoint_sha256": values[0],
                "conditioning": str(args.conditioning.resolve()),
                "conditioning_prompt_sha256": prompt_sha256,
                "sol_root": str(args.sol_root.resolve()),
                "diffusers_src": str(args.diffusers_src.resolve()),
                "mpa_root": str(ROOT),
                **source_provenance,
            },
            "transformer_build": build,
            "per_rank": per_rank,
            "rank_outputs_bit_identical": consistent,
            "candidates": {
                args.candidate: {
                    "policy": candidate,
                    "resident_denoise_wall_s": max(
                        item["resident_denoise_wall_s"] for item in per_rank
                    ),
                    "transformer_gpu_s_rank0": per_rank[0]["transformer_gpu_s"],
                    "dit_evals": per_rank[0]["dit_evals"],
                    "memory": {
                        "peak_allocated_mib_per_rank": [
                            item["peak_allocated_mib"] for item in per_rank
                        ]
                    },
                    "latent_artifact": str(latent_path),
                }
            },
        }
        _write_json(output_dir / "benchmark.json", results)
        _write_json(
            output_dir / "run_config.json",
            {
                "candidate": args.candidate,
                "candidate_policy": candidate,
                "world_size": 4,
                "ulysses_degree": 4,
                "decode": not args.no_decode,
                "prompt_id": args.prompt_id,
                "prompt_sha256": prompt_sha256,
                "prefix_tokens": prefix_tokens,
                "sequence_tokens": sequence_tokens,
                "video_shape": list(video_shape),
                "resolution": [args.width, args.height],
                "generated_frames": args.num_frames,
            },
        )

    for undo in reversed(cleanup):
        undo()
    # Python keeps a ``for`` target alive after the loop.  The official-Sol
    # uninstall closure captures ``transformer``; retaining that last target
    # therefore kept the full DiT resident and made the following VAE load OOM.
    del undo
    del timer, cleanup, attention
    del state, run_inputs, denoise_pipe, transformer, prompt_embeds, text_token_tags
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()

    if rank != 0:
        return 0
    assert results is not None
    if not args.no_decode:
        bench._decode_and_export(
            {args.candidate: artifact},
            results,
            output_dir,
            args.candidate,
        )
        decoded = Path(results["candidates"][args.candidate]["decode"]["output"])
        shutil.copy2(decoded, output_dir / "out.mp4")
    results["status"] = "complete"
    _write_json(output_dir / "benchmark.json", results)
    return 0


def main() -> int:
    args = _parser().parse_args()
    default_conditioning, _ = PROMPTS[args.prompt_id]
    if args.conditioning is None:
        args.conditioning = default_conditioning
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if (
        args.height <= 0
        or args.width <= 0
        or args.height % 32
        or args.width % 32
        or args.num_frames != 124
    ):
        raise ValueError(
            "height/width must be positive multiples of 32 and the current "
            "quality runner requires 124 generated frames"
        )
    if args.decode_only and args.no_decode:
        raise ValueError("--decode-only and --no-decode are mutually exclusive")
    if args.steps != 50 and not args.no_decode:
        raise ValueError("non-50-step smokes require --no-decode")
    candidate = _resolved_candidate(
        args.candidate,
        args.fp8_ratio,
        args.fp16_ratio,
    )
    if args.print_config:
        print(json.dumps({"candidate": args.candidate, "policy": candidate}, indent=2))
        return 0
    _configure_environment(args)
    if args.decode_only:
        return _decode_existing(args)
    return _main_distributed(args, candidate)


if __name__ == "__main__":
    raise SystemExit(main())
