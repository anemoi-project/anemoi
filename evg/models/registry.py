from __future__ import annotations

from collections.abc import Iterable

from evg.models.catalog import BUILTIN_MODEL_SPECS
from evg.models.specs import ModelSpec
from evg.types import TaskType


class ModelRegistry:
    def __init__(self, specs: Iterable[ModelSpec] = ()):
        self._specs: dict[str, ModelSpec] = {}
        self._aliases: dict[str, str] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        canonical = self._normalize(spec.id)
        if canonical in self._specs:
            raise ValueError(f"Model '{spec.id}' is already registered")
        self._specs[canonical] = spec
        for alias in (spec.name, *spec.aliases):
            self._aliases[self._normalize(alias)] = canonical

    def get(self, model_id: str) -> ModelSpec:
        key = self._normalize(model_id)
        canonical = self._aliases.get(key, key)
        try:
            return self._specs[canonical]
        except KeyError as exc:
            known = ", ".join(spec.id for spec in self.list())
            raise KeyError(f"Unknown model '{model_id}'. Known models: {known}") from exc

    def list(self, task: TaskType | None = None) -> tuple[ModelSpec, ...]:
        specs = tuple(self._specs[key] for key in sorted(self._specs))
        if task is None:
            return specs
        return tuple(spec for spec in specs if spec.supports(task))

    @staticmethod
    def _normalize(value: str) -> str:
        return value.strip().lower().replace("_", "-").replace(" ", "-")


_REGISTRY = ModelRegistry(BUILTIN_MODEL_SPECS)


def get_model_registry() -> ModelRegistry:
    return _REGISTRY
