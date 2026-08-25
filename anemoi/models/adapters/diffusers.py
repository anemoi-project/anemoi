from __future__ import annotations

from pathlib import Path
from typing import Any

from anemoi.models.adapters.base import AdapterError, VideoModelAdapter
from anemoi.types import GeneratedArtifact, GenerationRequest


class DiffusersVideoAdapter(VideoModelAdapter):
    backend = "diffusers"
    install_extras = ("runtime",)
    entrypoint = "diffusers.DiffusionPipeline"
    can_execute = True

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._pipe: Any | None = None

    def load(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        if self.variant.hf_repo_id is None:
            raise AdapterError(f"{self.spec.id}:{self.variant.id} has no Hugging Face repo id")

        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise AdapterError(
                "Diffusers runtime dependencies are missing. Install them with "
                "'pip install -e .[runtime]'."
            ) from exc

        dtype = getattr(torch, self.config.precision.weight_dtype, None)
        if dtype is None:
            raise AdapterError(f"Unknown torch dtype '{self.config.precision.weight_dtype}'")

        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.config.device == "cuda":
            kwargs["device_map"] = "cuda"
        if self.variant.hf_subfolder:
            kwargs["subfolder"] = self.variant.hf_subfolder

        self._pipe = DiffusionPipeline.from_pretrained(self.variant.hf_repo_id, **kwargs)
        if self.config.device != "cuda":
            self._pipe.to(self.config.device)
        return self._pipe

    def generate(self, request: GenerationRequest) -> GeneratedArtifact:
        self.validate_request(request)
        pipe = self.load()

        kwargs: dict[str, Any] = {"prompt": request.prompt}
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.num_frames:
            kwargs["num_frames"] = request.num_frames
        if request.height:
            kwargs["height"] = request.height
        if request.width:
            kwargs["width"] = request.width
        if request.fps:
            kwargs["fps"] = request.fps
        if request.seed is not None:
            kwargs["generator"] = self._make_generator(request.seed)

        result = pipe(**kwargs)
        frames = self._extract_frames(result)
        if frames is None:
            raise AdapterError(
                "The Diffusers pipeline returned an unsupported output shape. "
                "Add a family-specific adapter to map this output."
            )

        self._export_video(frames, request.output, request.fps or self.variant.default_fps or 24)
        return GeneratedArtifact(
            path=request.output,
            task=request.task,
            model=self.spec.id,
            variant=self.variant.id,
            metadata={"backend": self.backend, "hf_repo_id": self.variant.hf_repo_id},
        )

    def _make_generator(self, seed: int) -> Any:
        import torch

        device = self.config.device if self.config.device != "auto" else "cuda"
        return torch.Generator(device=device).manual_seed(seed)

    @staticmethod
    def _extract_frames(result: Any) -> Any | None:
        for attr in ("frames", "video", "videos"):
            value = getattr(result, attr, None)
            if value is not None:
                if isinstance(value, list) and value:
                    return value[0]
                return value
        images = getattr(result, "images", None)
        if images is not None:
            return images
        return None

    @staticmethod
    def _export_video(frames: Any, output: Path, fps: float) -> None:
        from diffusers.utils import export_to_video

        output.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(output), fps=int(fps), macro_block_size=1)
