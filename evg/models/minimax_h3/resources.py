"""Download and validate the external resources used by the MiniMax-H3 runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


DIFFUSERS_REPOSITORY = "https://github.com/huggingface/diffusers.git"
DIFFUSERS_COMMIT = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
MODEL_REPOSITORY = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "939557dc319dd91227e30195a763f272ba7f8765"
CHECKPOINT_REPOSITORY = "Comfy-Org/MiniMax-H3"
CHECKPOINT_REVISION = "014cd40f7e177756c6b2473c0d93b1c89a790dd2"
CHECKPOINT_FILENAME = (
    "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
)
CHECKPOINT_BYTES = 20_958_205_608
CHECKPOINT_SHA256 = (
    "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993"
)
MODEL_ALLOW_PATTERNS = (
    "modular_model_index.json",
    "transformer/config.json",
    "scheduler/*",
    "audio_scheduler/*",
    "vae/*",
    "audio_vae/*",
)


def _run_git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def prepare_diffusers(path: Path) -> Path:
    """Create or verify the pinned Diffusers checkout without replacing user data."""

    path = path.resolve()
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Diffusers target exists and is not a directory: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _run_git("clone", "--no-checkout", DIFFUSERS_REPOSITORY, str(path))
    if not (path / ".git").is_dir():
        raise RuntimeError(
            f"Diffusers target is not a Git checkout and will not be overwritten: {path}"
        )

    # ``git clone --no-checkout`` leaves an empty worktree whose index can look
    # like a staged deletion of every tracked file. It is safe to recover that
    # state because no user files exist outside ``.git``.
    empty_worktree = not any(entry.name != ".git" for entry in path.iterdir())
    dirty = _run_git("status", "--porcelain", cwd=path)
    if dirty and not empty_worktree:
        raise RuntimeError(
            f"Diffusers checkout has local changes and will not be modified: {path}"
        )
    if dirty:
        _run_git("reset", "--hard", "HEAD", cwd=path)
    head = _run_git("rev-parse", "HEAD", cwd=path)
    source_ready = (path / "src/diffusers/__init__.py").is_file()
    if head != DIFFUSERS_COMMIT:
        _run_git("fetch", "--depth=1", "origin", DIFFUSERS_COMMIT, cwd=path)
    if head != DIFFUSERS_COMMIT or not source_ready:
        _run_git("checkout", "--detach", DIFFUSERS_COMMIT, cwd=path)
    observed = _run_git("rev-parse", "HEAD", cwd=path)
    if observed != DIFFUSERS_COMMIT or not (path / "src/diffusers/__init__.py").is_file():
        raise RuntimeError(f"Diffusers revision mismatch: {observed}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_huggingface_resources(root: Path, token: str | None = None) -> tuple[Path, Path]:
    """Download only the model components required by denoise and decode."""

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; install requirements-minimax-h3.txt"
        ) from exc

    root = root.resolve()
    model_root = root / "model"
    checkpoint_root = root / "checkpoint"
    model_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        allow_patterns=list(MODEL_ALLOW_PATTERNS),
        local_dir=model_root,
        token=token,
    )
    checkpoint = Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPOSITORY,
            filename=CHECKPOINT_FILENAME,
            revision=CHECKPOINT_REVISION,
            local_dir=checkpoint_root,
            token=token,
        )
    ).resolve()
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise RuntimeError(
            f"compressed checkpoint has {checkpoint.stat().st_size} bytes; "
            f"expected {CHECKPOINT_BYTES}"
        )
    observed = _sha256(checkpoint)
    if observed != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"compressed checkpoint SHA256 {observed} != {CHECKPOINT_SHA256}"
        )
    required = (
        model_root / "modular_model_index.json",
        model_root / "transformer/config.json",
        model_root / "scheduler/scheduler_config.json",
        model_root / "audio_scheduler/scheduler_config.json",
        model_root / "vae/config.json",
        model_root / "audio_vae/config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"downloaded model snapshot is incomplete: {missing}")
    return model_root, checkpoint


def prepare_all(root: Path, token: str | None = None) -> dict[str, str]:
    diffusers = prepare_diffusers(root / "diffusers")
    model_root, checkpoint = prepare_huggingface_resources(root, token)
    return {
        "resource_root": str(root.resolve()),
        "diffusers": str(diffusers),
        "model_root": str(model_root),
        "checkpoint": str(checkpoint),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("EVG_MINIMAX_H3_ROOT", "models/minimax-h3")),
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args(argv)
    print(json.dumps(prepare_all(args.root, args.token), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_BYTES",
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_REVISION",
    "CHECKPOINT_SHA256",
    "DIFFUSERS_COMMIT",
    "MODEL_REVISION",
    "prepare_all",
]
