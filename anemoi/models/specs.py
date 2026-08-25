from __future__ import annotations

from dataclasses import dataclass, field

from anemoi.types import RuntimeKind, SupportStatus, TaskType


@dataclass(frozen=True)
class ModelSource:
    kind: str
    url: str


@dataclass(frozen=True)
class ModelVariant:
    id: str
    name: str
    tasks: tuple[TaskType, ...]
    hf_repo_id: str | None = None
    hf_subfolder: str | None = None
    modelscope_id: str | None = None
    parameter_count: str | None = None
    active_parameter_count: str | None = None
    resolutions: tuple[str, ...] = ()
    fps: tuple[int, ...] = ()
    default_resolution: str | None = None
    default_fps: int | None = None
    notes: tuple[str, ...] = ()
    extra: dict[str, str] = field(default_factory=dict)

    def supports(self, task: TaskType) -> bool:
        return task in self.tasks


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    organization: str
    adapter: str
    runtime: RuntimeKind
    status: SupportStatus
    variants: tuple[ModelVariant, ...]
    default_variant: str
    aliases: tuple[str, ...] = ()
    license_id: str | None = None
    description: str = ""
    sources: tuple[ModelSource, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def tasks(self) -> tuple[TaskType, ...]:
        seen: set[TaskType] = set()
        ordered: list[TaskType] = []
        for variant in self.variants:
            for task in variant.tasks:
                if task not in seen:
                    seen.add(task)
                    ordered.append(task)
        return tuple(ordered)

    def variant(self, variant_id: str | None = None) -> ModelVariant:
        requested = variant_id or self.default_variant
        for variant in self.variants:
            if variant.id == requested:
                return variant
        known = ", ".join(variant.id for variant in self.variants)
        raise KeyError(f"Unknown variant '{requested}' for {self.id}. Known variants: {known}")

    def supports(self, task: TaskType) -> bool:
        return any(variant.supports(task) for variant in self.variants)
