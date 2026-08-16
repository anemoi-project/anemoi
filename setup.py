"""Build package-private CUDA extensions for the active PyTorch runtime.

The standard Python extension suffix prevents loading a binary compiled for a
different interpreter ABI.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent


def _requested_components() -> set[str]:
    value = os.environ.get("MPA_BUILD_COMPONENTS", "sm89")
    requested = {item.strip() for item in value.split(",") if item.strip()}
    aliases = {"attention": "sm89"}
    known = {"sm89", *aliases}
    unknown = requested - known
    if unknown:
        raise RuntimeError(
            "MPA_BUILD_COMPONENTS contains unknown component(s): "
            + ", ".join(sorted(unknown))
        )
    return {aliases.get(component, component) for component in requested}


def _extensions():
    if os.environ.get("MPA_SKIP_CUDA_BUILD", "0").upper() in {
        "1",
        "TRUE",
        "YES",
    }:
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise RuntimeError(
            "mixed-precision-attention builds against the PyTorch wheel in "
            "the target runtime. Install PyTorch first, then use "
            "`python -m pip install . --no-build-isolation`."
        ) from exc

    components = _requested_components()
    modules = []

    if "sm89" in components:
        attention = ROOT / "csrc" / "attention" / "cuda"
        sm89 = attention / "sm89"
        common = attention / "common"
        modules.append(
            CUDAExtension(
                name="evg.layers.attention.mpa._cuda_attention",
                sources=[
                    str(sm89 / "bindings.cpp"),
                    str(sm89 / "k64_attention_host.cu"),
                    str(sm89 / "value_preprocess.cu"),
                    str(sm89 / "raster_preprocess.cu"),
                    str(sm89 / "output_assembly.cu"),
                    str(sm89 / "instantiations" / "inst_k64_d128.cu"),
                ],
                include_dirs=[
                    str(attention),
                    str(attention.parent),
                    str(sm89),
                    str(common),
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "--use_fast_math",
                        "-lineinfo",
                        "-Xptxas=-v",
                        "-Xcompiler",
                        "-include,cassert",
                    ],
                },
            )
        )

    return modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = _extensions()
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
