from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evg.config import AttentionConfig, EngineConfig, SparsityScheduleConfig
from evg.engine import EVGEngine
from evg.layers.attention.draft_attention import DraftAttention, DraftAttentionConfig
from evg.layers.attention.presets import PRESETS, get_preset
from evg.layers.attention.sparse_attention import dense_attention_reference
from evg.models import ModelSpec, ModelVariant, get_model_registry
from evg.models.adapters import AdapterError
from evg.types import GenerationRequest, TaskType


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AdapterError as exc:
        print(f"error: {exc}")
        return 2
    except KeyError as exc:
        print(f"error: {exc}")
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evg", description="Efficient Visual Generation")
    subparsers = parser.add_subparsers(required=True)

    list_cmd = subparsers.add_parser("list-models", help="List built-in video model families")
    list_cmd.add_argument("--task", choices=[task.value for task in TaskType])
    list_cmd.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    list_cmd.set_defaults(func=_list_models)

    inspect_cmd = subparsers.add_parser("inspect", help="Inspect a model family or variant")
    inspect_cmd.add_argument("model")
    inspect_cmd.add_argument("--variant")
    inspect_cmd.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    inspect_cmd.set_defaults(func=_inspect_model)

    generate_cmd = subparsers.add_parser("generate", help="Generate or dry-run a video request")
    generate_cmd.add_argument("--model", required=True)
    generate_cmd.add_argument("--variant")
    generate_cmd.add_argument("--task", default=TaskType.TEXT_TO_VIDEO.value)
    generate_cmd.add_argument("--prompt", required=True)
    generate_cmd.add_argument("--output", required=True)
    generate_cmd.add_argument("--negative-prompt")
    generate_cmd.add_argument("--seed", type=int)
    generate_cmd.add_argument("--width", type=int)
    generate_cmd.add_argument("--height", type=int)
    generate_cmd.add_argument("--num-frames", type=int)
    generate_cmd.add_argument("--fps", type=float)
    generate_cmd.add_argument("--device", default="cuda")
    generate_cmd.add_argument("--offload", action="store_true")
    generate_cmd.add_argument("--attention-backend", default="dense")
    generate_cmd.add_argument(
        "--sparsity-schedule",
        type=Path,
        help="JSON schedule with dense-step, per-step, and per-layer sparsity settings",
    )
    generate_cmd.add_argument("--dense-step-fraction", type=float, default=0.0)
    generate_cmd.add_argument("--sparsity-ratio", type=float, default=0.9)
    generate_cmd.add_argument("--pool-h", type=int, default=8)
    generate_cmd.add_argument("--pool-w", type=int, default=16)
    generate_cmd.add_argument("--dry-run", action="store_true")
    generate_cmd.set_defaults(func=_generate)

    smoke_cmd = subparsers.add_parser(
        "draft-attn-smoke",
        help="Run a synthetic Draft Attention sparse-attention smoke test",
    )
    smoke_cmd.add_argument("--preset", choices=sorted(PRESETS), required=True)
    smoke_cmd.add_argument("--full-shape", action="store_true")
    smoke_cmd.add_argument("--device", default="auto")
    smoke_cmd.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    smoke_cmd.add_argument("--backend", default="auto", choices=("auto", "torch", "triton"))
    smoke_cmd.add_argument("--seed", type=int, default=0)
    smoke_cmd.add_argument("--batch-size", type=int, default=1)
    smoke_cmd.add_argument("--sparsity-ratio", type=float)
    smoke_cmd.add_argument("--heads", type=int)
    smoke_cmd.add_argument("--head-dim", type=int)
    smoke_cmd.add_argument("--draft-q-chunk-size", type=int, default=64)
    smoke_cmd.add_argument("--draft-k-chunk-size", type=int, default=64)
    smoke_cmd.add_argument("--compare-dense", action="store_true")
    smoke_cmd.add_argument(
        "--full-mask-check",
        action="store_true",
        help="Force sparsity_ratio=0 and compare Draft Attention to dense attention",
    )
    smoke_cmd.set_defaults(func=_draft_attn_smoke)

    return parser


def _list_models(args: argparse.Namespace) -> int:
    task = TaskType(args.task) if args.task else None
    specs = get_model_registry().list(task=task)
    if args.json:
        print(json.dumps([_spec_to_dict(spec) for spec in specs], indent=2))
        return 0

    rows = []
    for spec in specs:
        rows.append(
            (
                spec.id,
                spec.default_variant,
                str(spec.runtime),
                str(spec.status),
                ", ".join(task.value for task in spec.tasks),
            )
        )
    _print_table(("model", "default", "runtime", "status", "tasks"), rows)
    return 0


def _inspect_model(args: argparse.Namespace) -> int:
    spec = get_model_registry().get(args.model)
    variant = spec.variant(args.variant) if args.variant else None
    payload = _spec_to_dict(spec)
    if variant:
        payload = {"model": payload, "selected_variant": _variant_to_dict(variant)}

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{spec.name} ({spec.id})")
    print(f"organization: {spec.organization}")
    print(f"runtime: {spec.runtime}")
    print(f"status: {spec.status}")
    print(f"default variant: {spec.default_variant}")
    print(f"tasks: {', '.join(task.value for task in spec.tasks)}")
    if spec.license_id:
        print(f"license: {spec.license_id}")
    if spec.description:
        print(f"description: {spec.description}")
    if spec.sources:
        print("sources:")
        for source in spec.sources:
            print(f"  {source.kind}: {source.url}")
    print("variants:")
    for item in spec.variants:
        marker = "*" if item.id == spec.default_variant else "-"
        print(f"  {marker} {item.id}: {item.name}")
        print(f"    tasks: {', '.join(task.value for task in item.tasks)}")
        if item.hf_repo_id:
            location = item.hf_repo_id
            if item.hf_subfolder:
                location = f"{location}/{item.hf_subfolder}"
            print(f"    hf: {location}")
    return 0


def _generate(args: argparse.Namespace) -> int:
    task = TaskType(args.task)
    schedule_enabled = args.attention_backend == "evg-draft"
    if args.sparsity_schedule is not None:
        schedule = SparsityScheduleConfig.from_json_file(
            args.sparsity_schedule,
            enabled=schedule_enabled,
        )
    else:
        schedule = SparsityScheduleConfig(
            enabled=schedule_enabled,
            dense_step_fraction=args.dense_step_fraction,
            default_sparsity=args.sparsity_ratio,
        )
    attention = AttentionConfig(
        backend=args.attention_backend,
        pool_h=args.pool_h,
        pool_w=args.pool_w,
        schedule=schedule,
    )
    config = EngineConfig(
        model=args.model,
        variant=args.variant,
        device=args.device,
        offload=args.offload,
        attention=attention,
    )
    engine = EVGEngine(config)
    request = GenerationRequest(
        model=args.model,
        variant=args.variant,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output=Path(args.output),
        task=task,
        seed=args.seed,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
    )

    if args.dry_run:
        plan = engine.plan()
        print(json.dumps(asdict(plan), indent=2))
        return 0

    artifact = engine.generate(request)
    print(str(artifact.path))
    return 0


def _draft_attn_smoke(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError as exc:
        raise AdapterError("PyTorch is required for draft-attn-smoke") from exc

    preset = get_preset(args.preset, full_shape=args.full_shape)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = getattr(torch, args.dtype)
    if device == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    sparsity_ratio = preset.sparsity_ratio if args.sparsity_ratio is None else args.sparsity_ratio
    if args.full_mask_check:
        sparsity_ratio = 0.0
        args.compare_dense = True

    num_heads = args.heads or preset.num_heads
    head_dim = args.head_dim or preset.head_dim
    seq_len = preset.visual_len + preset.text_len
    shape = (args.batch_size, seq_len, num_heads, head_dim)

    q = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(*shape, device=device, dtype=dtype)
    v = torch.randn(*shape, device=device, dtype=dtype)

    config = DraftAttentionConfig(
        latent_h=preset.latent_h,
        latent_w=preset.latent_w,
        visual_len=preset.visual_len,
        text_len=preset.text_len,
        pool_h=preset.pool_h,
        pool_w=preset.pool_w,
        sparsity_ratio=sparsity_ratio,
        draft_q_chunk_size=args.draft_q_chunk_size,
        draft_k_chunk_size=args.draft_k_chunk_size,
        backend=args.backend,
    )
    attention = DraftAttention(config)

    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    output, debug = attention(q, k, v, return_debug=True)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    payload: dict[str, Any] = {
        "preset": preset.name,
        "full_shape": args.full_shape,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": {
            "batch": args.batch_size,
            "seq_len": seq_len,
            "visual_len": preset.visual_len,
            "text_len": preset.text_len,
            "heads": num_heads,
            "head_dim": head_dim,
            "latent_h": preset.latent_h,
            "latent_w": preset.latent_w,
            "frames": preset.num_frames,
        },
        "sparsity_ratio": sparsity_ratio,
        "output_shape": list(output.shape),
        "elapsed_sec": elapsed,
        "draft_density": debug.draft_density,
        "sequence_density": debug.sequence_density,
        "q_block_size": debug.q_block_size,
        "k_block_size": debug.k_block_size,
        "draft_blocks": [debug.draft_q_blocks, debug.draft_k_blocks],
        "sequence_blocks": [debug.sequence_q_blocks, debug.sequence_k_blocks],
        "backend": debug.backend,
    }

    if args.compare_dense:
        dense = dense_attention_reference(q, k, v)
        diff = (output.float() - dense.float()).abs()
        payload["dense_compare"] = {
            "max_abs_error": float(diff.max().item()),
            "mean_abs_error": float(diff.mean().item()),
        }

    print(json.dumps(payload, indent=2))
    return 0


def _spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": spec.name,
        "organization": spec.organization,
        "adapter": spec.adapter,
        "runtime": str(spec.runtime),
        "status": str(spec.status),
        "default_variant": spec.default_variant,
        "aliases": list(spec.aliases),
        "license_id": spec.license_id,
        "description": spec.description,
        "tasks": [task.value for task in spec.tasks],
        "sources": [asdict(source) for source in spec.sources],
        "notes": list(spec.notes),
        "variants": [_variant_to_dict(variant) for variant in spec.variants],
    }


def _variant_to_dict(variant: ModelVariant) -> dict[str, Any]:
    return {
        "id": variant.id,
        "name": variant.name,
        "tasks": [task.value for task in variant.tasks],
        "hf_repo_id": variant.hf_repo_id,
        "hf_subfolder": variant.hf_subfolder,
        "modelscope_id": variant.modelscope_id,
        "parameter_count": variant.parameter_count,
        "active_parameter_count": variant.active_parameter_count,
        "resolutions": list(variant.resolutions),
        "fps": list(variant.fps),
        "default_resolution": variant.default_resolution,
        "default_fps": variant.default_fps,
        "notes": list(variant.notes),
        "extra": dict(variant.extra),
    }


def _print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


if __name__ == "__main__":
    raise SystemExit(main())
