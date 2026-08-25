from __future__ import annotations

from anemoi.config import EngineConfig
from anemoi.models.adapters.base import AdapterError, AdapterRuntimePlan, VideoModelAdapter
from anemoi.models.adapters.families import (
    BerniniAdapter,
    Cosmos3Adapter,
    LingBotVideoAdapter,
    LongCatVideoAdapter,
    MiniMaxH3Adapter,
    SkyReelsV3Adapter,
    Wan22Adapter,
)
from anemoi.models.specs import ModelSpec


ADAPTERS: dict[str, type[VideoModelAdapter]] = {
    "minimax-h3": MiniMaxH3Adapter,
    "wan2.2": Wan22Adapter,
    "lingbot-video": LingBotVideoAdapter,
    "longcat-video": LongCatVideoAdapter,
    "cosmos3": Cosmos3Adapter,
    "skyreels-v3": SkyReelsV3Adapter,
    "bernini": BerniniAdapter,
}


def create_adapter(
    spec: ModelSpec, variant_id: str | None, config: EngineConfig
) -> VideoModelAdapter:
    variant = spec.variant(variant_id)
    adapter_cls = ADAPTERS.get(spec.adapter)
    if adapter_cls is None:
        raise AdapterError(f"No adapter registered for '{spec.adapter}'")
    return adapter_cls(spec, variant, config)


__all__ = ["AdapterError", "AdapterRuntimePlan", "VideoModelAdapter", "create_adapter"]
