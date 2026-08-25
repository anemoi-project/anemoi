from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionBackend:
    name: str
    supports_training_free: bool
    description: str
    implementation: str = "torch"


class AttentionBackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, AttentionBackend] = {}

    def register(self, backend: AttentionBackend) -> None:
        self._backends[backend.name] = backend

    def get(self, name: str) -> AttentionBackend:
        return self._backends[name]

    def list(self) -> tuple[AttentionBackend, ...]:
        return tuple(self._backends[key] for key in sorted(self._backends))


DEFAULT_ATTENTION_BACKENDS = AttentionBackendRegistry()
DEFAULT_ATTENTION_BACKENDS.register(
    AttentionBackend(
        name="dense",
        supports_training_free=True,
        description="Baseline dense attention using the model's upstream implementation.",
    )
)
DEFAULT_ATTENTION_BACKENDS.register(
    AttentionBackend(
        name="draft",
        supports_training_free=True,
        description="Draft Attention with blockwise draft-map construction and sparse attention.",
        implementation="torch-reference/triton-forward",
    )
)
DEFAULT_ATTENTION_BACKENDS.register(
    AttentionBackend(
        name="draft-sparse",
        supports_training_free=True,
        description="Alias for Draft Attention sparse inference.",
        implementation="torch-reference/triton-forward",
    )
)
