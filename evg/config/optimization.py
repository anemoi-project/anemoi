from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


IndexSelector = tuple[int, ...] | None


def _validate_sparsity(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return value


def _parse_index_selector(value: Any, name: str) -> IndexSelector:
    if value is None or value == "*":
        return None
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, str):
        values = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"{name} range must be ascending: {part}")
                values.extend(range(start, end + 1))
            else:
                values.append(int(part))
    elif isinstance(value, Sequence):
        values = [int(index) for index in value]
    else:
        raise TypeError(f"{name} must be '*', an integer, a range string, or a sequence")

    if not values:
        raise ValueError(f"{name} cannot be empty; use '*' to select all")
    if any(index < 0 for index in values):
        raise ValueError(f"{name} indices must be non-negative")
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class SparsityRule:
    """Override sparsity for selected diffusion steps and model layers."""

    sparsity: float
    steps: IndexSelector = None
    layers: IndexSelector = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sparsity", _validate_sparsity(self.sparsity, "sparsity"))
        object.__setattr__(self, "steps", _parse_index_selector(self.steps, "steps"))
        object.__setattr__(self, "layers", _parse_index_selector(self.layers, "layers"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SparsityRule":
        unknown = set(data) - {"sparsity", "steps", "layers"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown sparsity rule fields: {names}")
        if "sparsity" not in data:
            raise ValueError("Each sparsity rule requires a 'sparsity' value")
        return cls(
            sparsity=data["sparsity"],
            steps=data.get("steps"),
            layers=data.get("layers"),
        )


@dataclass(frozen=True)
class ResolvedSparsitySchedule:
    """Validated sparsity ratios indexed by diffusion step and model layer."""

    ratios: tuple[tuple[float, ...], ...]
    dense_steps: int

    @property
    def total_steps(self) -> int:
        return len(self.ratios)

    @property
    def num_layers(self) -> int:
        return len(self.ratios[0])

    def sparsity_for(self, step_index: int, layer_index: int) -> float:
        if not 0 <= step_index < self.total_steps:
            raise IndexError(f"step_index {step_index} is outside [0, {self.total_steps})")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} is outside [0, {self.num_layers})")
        return self.ratios[step_index][layer_index]

    def sparsities_for_step(self, step_index: int) -> tuple[float, ...]:
        if not 0 <= step_index < self.total_steps:
            raise IndexError(f"step_index {step_index} is outside [0, {self.total_steps})")
        return self.ratios[step_index]

    def is_dense_step(self, step_index: int) -> bool:
        return all(sparsity == 0.0 for sparsity in self.sparsities_for_step(step_index))


@dataclass(frozen=True)
class SparsityScheduleConfig:
    """Flexible sparse-attention policy for diffusion inference.

    Resolution order is default sparsity, optional full matrix, then ordered
    rules. The initial dense-step fraction is applied last and always wins.
    """

    enabled: bool = False
    dense_step_fraction: float = 0.0
    default_sparsity: float = 0.0
    rules: tuple[SparsityRule, ...] = field(default_factory=tuple)
    sparsity_matrix: tuple[tuple[float, ...], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.dense_step_fraction <= 1.0:
            raise ValueError("dense_step_fraction must be in [0, 1]")
        object.__setattr__(
            self,
            "default_sparsity",
            _validate_sparsity(self.default_sparsity, "default_sparsity"),
        )
        normalized_rules = tuple(
            rule if isinstance(rule, SparsityRule) else SparsityRule.from_dict(rule)
            for rule in self.rules
        )
        object.__setattr__(self, "rules", normalized_rules)
        normalized_matrix = tuple(
            tuple(_validate_sparsity(value, "sparsity_matrix value") for value in row)
            for row in self.sparsity_matrix
        )
        object.__setattr__(self, "sparsity_matrix", normalized_matrix)

    @property
    def dense_fraction(self) -> float:
        """Compatibility alias for early integrations."""

        return self.dense_step_fraction

    @property
    def sparse_ratio(self) -> float:
        """Compatibility alias for early integrations."""

        return self.default_sparsity

    def dense_step_count(self, total_steps: int) -> int:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        return min(total_steps, int(math.ceil(total_steps * self.dense_step_fraction)))

    def dense_steps(self, total_steps: int) -> int:
        """Compatibility alias for early integrations."""

        return self.dense_step_count(total_steps)

    def resolve(self, total_steps: int, num_layers: int) -> ResolvedSparsitySchedule:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        default = self.default_sparsity if self.enabled else 0.0
        ratios = [[default for _ in range(num_layers)] for _ in range(total_steps)]

        if self.enabled and self.sparsity_matrix:
            if len(self.sparsity_matrix) != total_steps:
                raise ValueError(
                    "sparsity_matrix row count must equal total_steps: "
                    f"{len(self.sparsity_matrix)} != {total_steps}"
                )
            for step_index, row in enumerate(self.sparsity_matrix):
                if len(row) != num_layers:
                    raise ValueError(
                        f"sparsity_matrix row {step_index} must contain {num_layers} layers, "
                        f"got {len(row)}"
                    )
                ratios[step_index] = list(row)

        if self.enabled:
            for rule_index, rule in enumerate(self.rules):
                steps = range(total_steps) if rule.steps is None else rule.steps
                layers = range(num_layers) if rule.layers is None else rule.layers
                for step_index in steps:
                    if not 0 <= step_index < total_steps:
                        raise ValueError(
                            f"rule {rule_index} step {step_index} is outside [0, {total_steps})"
                        )
                    for layer_index in layers:
                        if not 0 <= layer_index < num_layers:
                            raise ValueError(
                                f"rule {rule_index} layer {layer_index} is outside "
                                f"[0, {num_layers})"
                            )
                        ratios[step_index][layer_index] = rule.sparsity

        dense_steps = self.dense_step_count(total_steps) if self.enabled else total_steps
        for step_index in range(dense_steps):
            ratios[step_index] = [0.0] * num_layers

        return ResolvedSparsitySchedule(
            ratios=tuple(tuple(row) for row in ratios),
            dense_steps=dense_steps,
        )

    def sparsity_for_step(self, step_index: int, total_steps: int) -> float:
        """Return the scalar schedule value used by early uniform integrations."""

        resolved = self.resolve(total_steps=total_steps, num_layers=1)
        return resolved.sparsity_for(step_index, 0)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, enabled: bool | None = None
    ) -> "SparsityScheduleConfig":
        aliases = {
            "dense_fraction": "dense_step_fraction",
            "sparse_ratio": "default_sparsity",
            "matrix": "sparsity_matrix",
        }
        normalized = {aliases.get(name, name): value for name, value in data.items()}
        known = {
            "enabled",
            "dense_step_fraction",
            "default_sparsity",
            "rules",
            "sparsity_matrix",
        }
        unknown = set(normalized) - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown sparsity schedule fields: {names}")
        if enabled is not None:
            normalized["enabled"] = enabled
        return cls(**normalized)

    @classmethod
    def from_json_file(
        cls, path: str | Path, *, enabled: bool | None = None
    ) -> "SparsityScheduleConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ValueError("Sparsity schedule JSON must contain an object")
        return cls.from_dict(data, enabled=enabled)
