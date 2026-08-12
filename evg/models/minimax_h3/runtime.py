"""Small runtime support shared by the three MiniMax-H3 demo candidates."""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch


def _model_root() -> Path:
    value = os.environ.get("H3_MODEL_ROOT")
    if not value:
        raise RuntimeError("H3_MODEL_ROOT is not configured")
    return Path(value)


class EvalTimer:
    """CUDA-event timing for every transformer evaluation."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.original = module.forward
        self.spans: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def install(self) -> None:
        original = self.original
        spans = self.spans

        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = original(*args, **kwargs)
            end.record()
            spans.append((start, end))
            return output

        self.module.forward = wrapped

    def remove(self) -> None:
        self.module.forward = self.original

    def samples_ms(self) -> list[float]:
        torch.cuda.synchronize()
        return [float(start.elapsed_time(end)) for start, end in self.spans]


def install_streamed_weight_only_fp8() -> dict[str, Any]:
    """Retain weight-only FP8 matrices and materialize BF16 per linear call."""

    import fp8_linear

    cls = fp8_linear.Fp8Linear
    if getattr(cls, "_mpa_streamed_weight_only_installed", False):
        return {"installed": True, "already_installed": True}
    original_init = cls.__init__
    original_forward = cls.forward

    def init(
        self,
        weight,
        weight_scale,
        input_scale,
        bias,
        compute_dtype=torch.bfloat16,
        quantizer="eager",
    ):
        if input_scale is not None:
            original_init(
                self,
                weight,
                weight_scale,
                input_scale,
                bias,
                compute_dtype,
                quantizer,
            )
            self._mpa_streamed_weight_only = False
            return
        torch.nn.Module.__init__(self)
        self.compute_dtype = compute_dtype
        self.quantized_activations = False
        self._quantize = fp8_linear.QUANTIZERS[quantizer]
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer(
            "weight_scale", weight_scale.to(compute_dtype), persistent=False
        )
        self.input_scale = None
        self.register_buffer(
            "bias", None if bias is None else bias.to(compute_dtype), persistent=False
        )
        self._mpa_streamed_weight_only = True

    def forward(self, x):
        if not getattr(self, "_mpa_streamed_weight_only", False):
            return original_forward(self, x)
        weight = self.weight.to(self.compute_dtype) * self.weight_scale
        return torch.nn.functional.linear(x.to(self.compute_dtype), weight, self.bias)

    cls.__init__ = init
    cls.forward = forward
    cls._mpa_streamed_weight_only_installed = True
    return {
        "installed": True,
        "already_installed": False,
        "semantics": "same BF16 cast and scale multiply as eager dequantization",
        "residency": "FP8 weights stay on CUDA; temporary BF16 is per call",
    }


def build_denoise_pipe(transformer: torch.nn.Module):
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
        MiniMaxH3Blocks,
    )

    blocks = MiniMaxH3Blocks()
    for name in ("text_encoder", "vae_encoder", "decode"):
        blocks.sub_blocks.pop(name)
    pipe = blocks.init_pipeline(_model_root())
    pipe.update_components(transformer=transformer)
    pipe.load_components(names=["scheduler", "audio_scheduler"])
    pipe.set_progress_bar_config(disable=False)
    return pipe


def run_inputs(
    prompt_embeds: torch.Tensor,
    text_token_tags: torch.Tensor,
    args,
) -> dict[str, Any]:
    return {
        "prompt_embeds": prompt_embeds.to("cuda", dtype=torch.bfloat16),
        "text_token_tags": text_token_tags,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "generator": torch.Generator().manual_seed(args.seed),
    }


def _probe_video(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(video=0))
        return {
            "frames": frames,
            "fps": float(stream.average_rate),
            "width": stream.width,
            "height": stream.height,
            "audio_streams": len(container.streams.audio),
        }


def decode_and_export(
    latent_artifacts: dict[str, dict[str, Any]],
    results: dict[str, Any],
    output_dir: Path,
    _quality_baseline: str,
) -> None:
    """Decode a single candidate; cross-candidate metrics use compare_outputs."""

    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
        MiniMaxH3DecodeStep,
    )
    from diffusers.utils.export_utils import encode_video

    if len(latent_artifacts) != 1:
        raise ValueError("the compact demo decoder accepts one candidate per process")
    decode_pipe = MiniMaxH3DecodeStep().init_pipeline(_model_root())
    decode_pipe.load_components(names=["vae", "audio_vae"])
    decode_pipe.vae.to("cuda")
    decode_pipe.audio_vae.to("cuda")
    torch.cuda.synchronize()
    resident_mib = torch.cuda.memory_allocated() / 2**20

    name, artifact = next(iter(latent_artifacts.items()))
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall0 = time.perf_counter()
    start.record()
    state = decode_pipe(
        latents=artifact["latents"].to("cuda"),
        audio_latents=artifact["audio_latents"].to("cuda"),
        num_condition_video_rows=artifact["num_condition_video_rows"],
        num_condition_audio_rows=artifact["num_condition_audio_rows"],
        num_latent_frames=artifact["num_latent_frames"],
        latent_height=artifact["latent_height"],
        latent_width=artifact["latent_width"],
        num_audio_latents=artifact["num_audio_latents"],
        output_type="pil",
    )
    end.record()
    torch.cuda.synchronize()
    frames = state.videos[0][:120]
    audio = state.audio[0].detach().cpu()
    sampling_rate = int(state.sampling_rate)
    trimmed_audio = audio[..., : 5 * sampling_rate]
    video_path = output_dir / f"{name}_{artifact['width']}x{artifact['height']}_5s_24fps.mp4"
    encode0 = time.perf_counter()
    encode_video(
        frames,
        fps=24,
        output_path=str(video_path),
        audio=trimmed_audio,
        audio_sample_rate=sampling_rate,
    )
    probe = _probe_video(video_path)
    if (
        probe["frames"] != 120
        or not math.isclose(probe["fps"], 24.0)
        or (probe["width"], probe["height"])
        != (artifact["width"], artifact["height"])
    ):
        raise RuntimeError(f"encoded media failed its contract: {probe}")
    results["candidates"][name]["decode"] = {
        "resident_mib": round(resident_mib, 1),
        "gpu_s": round(start.elapsed_time(end) / 1000.0, 6),
        "wall_s": round(time.perf_counter() - wall0, 6),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "encode_wall_s": round(time.perf_counter() - encode0, 6),
        "output": str(video_path),
        "delivered_frames": len(frames),
        "audio_sampling_rate": sampling_rate,
        "media_probe": probe,
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(results, indent=2, allow_nan=True) + "\n"
    )
    del state, frames, audio, trimmed_audio, decode_pipe
    gc.collect()
    torch.cuda.empty_cache()


# Private aliases keep the frozen runner call sites small and auditable.
_install_streamed_weight_only_fp8 = install_streamed_weight_only_fp8
_build_denoise_pipe = build_denoise_pipe
_run_inputs = run_inputs
_decode_and_export = decode_and_export
