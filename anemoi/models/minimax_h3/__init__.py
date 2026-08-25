"""MiniMax-H3 integration for ragged mixed attention."""

from __future__ import annotations

import os


# This package imports Torch below. Configure its CUDA allocator first so the
# 24 GiB H3 profile can reuse fragmented segments during Ulysses exchanges.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from .mpa_attention import H3MPAAttention, H3MPAConfig, install_layout_observer

__all__ = [
    "H3MPAAttention",
    "H3MPAConfig",
    "install_layout_observer",
]
