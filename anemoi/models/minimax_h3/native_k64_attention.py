"""Compatibility alias for the generic ragged MPA executor."""

import sys

from anemoi.layers.attention.mpa import executor

sys.modules[__name__] = executor
