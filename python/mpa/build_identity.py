"""Load the package-local native extension built for this Python ABI."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import sysconfig


_MODULES = {
    "router": "mpa._cuda_router",
    "attention": "mpa._cuda_attention",
}


def _extension_path(component: str) -> Path:
    try:
        module = _MODULES[component]
    except KeyError as exc:
        raise ValueError(f"unknown native extension: {component}") from exc
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python did not report an extension suffix")
    return Path(__file__).resolve().parent / f"{module.rsplit('.', 1)[1]}{suffix}"


def import_native_extension(component: str):
    """Import only the exact package artifact for the active interpreter."""

    module_name = _MODULES.get(component)
    if module_name is None:
        raise ValueError(f"unknown native extension: {component}")
    expected = _extension_path(component).resolve()
    if not expected.is_file():
        raise RuntimeError(
            f"{module_name} is not built for this Python ABI; run "
            f"scripts/build_{component}_cuda.sh"
        )
    existing = sys.modules.get(module_name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != expected:
            raise RuntimeError(f"preloaded {module_name} is not {expected}")
        return existing
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise RuntimeError(f"import resolution for {module_name} did not select {expected}")
    return importlib.import_module(module_name)


def resolve_mixed_attention_operator(name: str):
    """Resolve one operator registered in ``torch.ops.mixed_attention``."""

    if not isinstance(name, str) or not name:
        raise ValueError("operator name must be nonempty")
    import torch

    try:
        operation = getattr(torch.ops.mixed_attention, name).default
    except AttributeError as exc:
        raise RuntimeError(f"mixed_attention.{name} is not registered") from exc
    return operation


__all__ = ["import_native_extension", "resolve_mixed_attention_operator"]
