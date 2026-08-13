"""MiniMax-H3 integration for the SM89 mixed-attention demo."""

from __future__ import annotations

import os


# This package imports Torch below. Configure its CUDA allocator first so the
# 24 GiB H3 profile can reuse fragmented segments during Ulysses exchanges.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .mpa_attention import H3MPAAttention, H3MPAConfig, install_layout_observer
from .sm89_mainline import (
    SM89Adaptive2DPlan,
    SM89_PHYSICAL_K_TOKENS,
    SM89_VALIDATED_LOGICAL_TILES,
    select_sm89_adaptive_2d_plan,
)

__all__ = [
    "H3MPAAttention",
    "H3MPAConfig",
    "install_layout_observer",
    "SM89Adaptive2DPlan",
    "SM89_PHYSICAL_K_TOKENS",
    "SM89_VALIDATED_LOGICAL_TILES",
    "select_sm89_adaptive_2d_plan",
]
