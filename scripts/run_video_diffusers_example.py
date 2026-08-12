from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path


DEFAULT_PROMPT = (
    "In the style of Dunhuang sculptures, A graceful deity, playing a pipa, "
    "dances lightly in a museum, with flowing garments."
)

WAN_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small Diffusers video-generation example.")
    parser.add_argument("--model", choices=("wan2.2",), required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-root",
        default=os.environ.get("EVG_MODEL_ROOT"),
        help="Optional directory containing local model checkpoints",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME"),
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--cpu-offload", action="store_true", default=True)
    parser.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false")
    parser.add_argument("--latent-only", action="store_true")
    args = parser.parse_args()

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", str(Path(args.cache_dir) / "hub"))

    import torch
    from diffusers.utils import export_to_video

    torch_dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    started = time.perf_counter()

    frames = run_wan22(args, torch_dtype, generator)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.latent_only:
        torch.save(frames, output)
    else:
        export_to_video(frames, str(output), fps=args.fps)

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    else:
        peak_gb = 0.0
        reserved_gb = 0.0

    print(
        json.dumps(
            {
                "model": args.model,
                "output": str(output),
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "fps": args.fps,
                "seed": args.seed,
                "dtype": args.dtype,
                "cpu_offload": args.cpu_offload,
                "elapsed_sec": time.perf_counter() - started,
                "peak_allocated_gb": peak_gb,
                "peak_reserved_gb": reserved_gb,
            },
            indent=2,
        )
    )
    return 0


def run_wan22(args: argparse.Namespace, torch_dtype, generator):
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

    model_id = resolve_model_path(
        args.model_root,
        "Wan2.2-TI2V-5B-Diffusers",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    prepare_pipeline(pipe, args.cpu_offload)
    output = pipe(
        prompt=args.prompt,
        negative_prompt=WAN_NEGATIVE_PROMPT,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="latent" if args.latent_only else "np",
    )
    frames = output.frames[0]
    cleanup_pipeline(pipe)
    return frames


def prepare_pipeline(pipe, cpu_offload: bool) -> None:
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")


def cleanup_pipeline(pipe) -> None:
    import torch

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_model_path(model_root: str | None, local_name: str, repo_id: str) -> str:
    if model_root:
        local_path = Path(model_root) / local_name
        if local_path.exists():
            return str(local_path)
    return repo_id


if __name__ == "__main__":
    raise SystemExit(main())
