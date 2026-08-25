from __future__ import annotations

from dataclasses import dataclass

from anemoi.config import EngineConfig
from anemoi.models.specs import ModelSpec, ModelVariant
from anemoi.types import GeneratedArtifact, GenerationRequest


class AdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdapterRuntimePlan:
    model_id: str
    variant_id: str
    runtime: str
    backend: str
    install_extras: tuple[str, ...]
    entrypoint: str
    can_execute: bool
    notes: tuple[str, ...] = ()


class VideoModelAdapter:
    backend = "base"
    install_extras: tuple[str, ...] = ()
    entrypoint = "native Anemoi adapter"
    can_execute = False

    def __init__(self, spec: ModelSpec, variant: ModelVariant, config: EngineConfig):
        self.spec = spec
        self.variant = variant
        self.config = config

    def validate_request(self, request: GenerationRequest) -> None:
        if not self.variant.supports(request.task):
            supported = ", ".join(str(task) for task in self.variant.tasks)
            raise AdapterError(
                f"{self.spec.id}:{self.variant.id} does not support task "
                f"'{request.task}'. Supported tasks: {supported}"
            )

    def runtime_plan(self) -> AdapterRuntimePlan:
        notes = (*self.spec.notes, *self.variant.notes)
        return AdapterRuntimePlan(
            model_id=self.spec.id,
            variant_id=self.variant.id,
            runtime=str(self.spec.runtime),
            backend=self.backend,
            install_extras=self.install_extras,
            entrypoint=self.entrypoint,
            can_execute=self.can_execute,
            notes=notes,
        )

    def load(self) -> None:
        raise AdapterError(f"{self.backend} loading is not implemented yet")

    def generate(self, request: GenerationRequest) -> GeneratedArtifact:
        self.validate_request(request)
        raise AdapterError(
            f"{self.spec.name} has an Anemoi adapter contract, but the executable runner "
            "has not landed yet. Use 'anemoi inspect' to see the upstream model id."
        )
