from __future__ import annotations

from dataclasses import dataclass, field

from evg.config.optimization import SparsityScheduleConfig


@dataclass(frozen=True)
class AttentionConfig:
    backend: str = "dense"
    block_size: int | None = None
    pool_h: int = 8
    pool_w: int = 16
    latent_h: int | None = None
    latent_w: int | None = None
    visual_len: int | None = None
    text_len: int = 0
    draft_q_chunk_size: int = 64
    draft_k_chunk_size: int = 64
    sparse_q_block_size: int | None = None
    sparse_k_block_size: int | None = None
    sparse_backend: str = "auto"
    schedule: SparsityScheduleConfig = field(default_factory=SparsityScheduleConfig)
    extra: dict[str, str] = field(default_factory=dict)


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
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)
