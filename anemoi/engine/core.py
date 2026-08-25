from __future__ import annotations

from anemoi.config import EngineConfig
from anemoi.models import get_model_registry
from anemoi.models.adapters import AdapterRuntimePlan, VideoModelAdapter, create_adapter
from anemoi.types import GeneratedArtifact, GenerationRequest


class AnemoiEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.spec = get_model_registry().get(config.model)
        self.adapter: VideoModelAdapter = create_adapter(self.spec, config.variant, config)

    @classmethod
    def from_model(
        cls,
        model: str,
        variant: str | None = None,
        device: str = "cuda",
        offload: bool = False,
    ) -> "AnemoiEngine":
        return cls(EngineConfig(model=model, variant=variant, device=device, offload=offload))

    def plan(self) -> AdapterRuntimePlan:
        return self.adapter.runtime_plan()

    def generate(self, request: GenerationRequest) -> GeneratedArtifact:
        if request.model != self.config.model:
            request = GenerationRequest(
                model=self.config.model,
                variant=request.variant or self.config.variant,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                output=request.output,
                task=request.task,
                seed=request.seed,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                fps=request.fps,
                duration=request.duration,
                media=request.media,
                extra=request.extra,
            )
        return self.adapter.generate(request)
