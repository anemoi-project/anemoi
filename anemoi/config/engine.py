from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrecisionPolicy:
    """Future mixed-precision policy; the current acceleration path is BF16 sparse attention."""

    weight_dtype: str = "bfloat16"
    activation_dtype: str = "bfloat16"
    attention_dtype: str = "bfloat16"
    gemm_dtype: str | None = None
    allow_tf32: bool = True


@dataclass(frozen=True)
class EngineConfig:
    model: str
    variant: str | None = None
    device: str = "cuda"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    offload: bool = False
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)
