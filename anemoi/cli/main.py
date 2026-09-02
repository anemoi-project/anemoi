from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from anemoi.config import EngineConfig
from anemoi.engine import AnemoiEngine
from anemoi.models import ModelSpec, ModelVariant, get_model_registry
from anemoi.models.adapters import AdapterError
from anemoi.types import GenerationRequest, TaskType


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
    parser = argparse.ArgumentParser(prog="anemoi", description="Efficient Visual Generation")
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
    generate_cmd.add_argument("--dry-run", action="store_true")
    generate_cmd.set_defaults(func=_generate)

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
    config = EngineConfig(
        model=args.model,
        variant=args.variant,
        device=args.device,
        offload=args.offload,
    )
    engine = AnemoiEngine(config)
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
