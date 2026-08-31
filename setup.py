"""Build package-private CUDA extensions for the active PyTorch runtime.

The standard Python extension suffix prevents loading a binary compiled for a
different interpreter ABI.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent


def _source(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _requested_components() -> set[str]:
    value = os.environ.get("MPA_BUILD_COMPONENTS", "sm89")
    requested = {item.strip() for item in value.split(",") if item.strip()}
    aliases = {
        "attention": ("sm89",),
        "sm120": ("sm89", "sm120_q64"),
    }
    known = {"sm89", "sm120_q64", *aliases}
    unknown = requested - known
    if unknown:
        raise RuntimeError(
            "MPA_BUILD_COMPONENTS contains unknown component(s): " + ", ".join(sorted(unknown))
        )
    return {
        resolved for component in requested for resolved in aliases.get(component, (component,))
    }


def _extensions():
    if os.environ.get("MPA_SKIP_CUDA_BUILD", "0").upper() in {
        "1",
        "TRUE",
        "YES",
    }:
        return [], {}

    components = _requested_components()
    default_arch = "12.0a" if "sm120_q64" in components else "8.9"
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get(
        "MPA_CUDA_ARCH_LIST",
        os.environ.get(
            "MPA_TORCH_CUDA_ARCH_LIST",
            os.environ.get("TORCH_CUDA_ARCH_LIST", default_arch),
        ),
    )

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

    modules = []

    if "sm89" in components:
        attention = ROOT / "csrc" / "attention" / "cuda"
        sm89 = attention / "sm89"
        common = attention / "common"
        modules.append(
            CUDAExtension(
                name="anemoi.layers.attention.mpa._cuda_attention",
                sources=[
                    _source(sm89 / "bindings.cpp"),
                    _source(attention / "sm120" / "h3_draft_probability.cu"),
                    _source(sm89 / "h3_route_precision.cu"),
                    _source(sm89 / "k64_attention_host.cu"),
                    _source(sm89 / "value_preprocess.cu"),
                    _source(sm89 / "raster_preprocess.cu"),
                    _source(sm89 / "output_assembly.cu"),
                    _source(sm89 / "instantiations" / "inst_k64_d128.cu"),
                    _source(sm89 / "instantiations" / "inst_k64_d128_int8_dense.cu"),
                    _source(sm89 / "instantiations" / "inst_q128_k64_d128.cu"),
                    _source(sm89 / "instantiations" / "inst_q128_k64_d128_fp16.cu"),
                    _source(sm89 / "instantiations" / "inst_q128_k64_d128_int8.cu"),
                    _source(sm89 / "instantiations" / "inst_q128_k64_d64.cu"),
                    _source(sm89 / "instantiations" / "inst_q128_k64_d64_fp16.cu"),
                    _source(sm89 / "instantiations" / "inst_q128_k64_d64_int8.cu"),
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

    if "sm120_q64" in components:
        attention = ROOT / "csrc" / "attention" / "cuda"
        sm120 = attention / "sm120"
        common = attention / "common"
        modules.append(
            CUDAExtension(
                name="anemoi.layers.attention.mpa._cuda_sm120_q64",
                sources=[
                    _source(sm120 / "bindings.cpp"),
                    _source(sm120 / "h3_draft_probability.cu"),
                    _source(sm120 / "h3_route_precision.cu"),
                    _source(sm120 / "q64_attention_host.cu"),
                    _source(sm120 / "instantiations" / "inst_q64_k64_d128_fp16.cu"),
                    _source(sm120 / "instantiations" / "inst_q64_k64_d128_int8.cu"),
                    _source(sm120 / "instantiations" / "inst_q64_k64_d128_int8_fp16.cu"),
                    _source(sm120 / "instantiations" / "inst_q64_k64_d128_int8_dense.cu"),
                    _source(sm120 / "instantiations" / "inst_q128_k64_d128_fp16.cu"),
                    _source(sm120 / "instantiations" / "inst_q128_k64_d128_int8.cu"),
                    _source(sm120 / "instantiations" / "inst_q128_k64_d128_int8_dense.cu"),
                    _source(sm120 / "instantiations" / "inst_q128_k64_d128_mxfp8.cu"),
                    _source(sm120 / "instantiations" / "inst_q128_k64_d128_mxfp8_compact.cu"),
                    _source(sm120 / "instantiations" / "inst_q128_k64_d128_nvfp4.cu"),
                    _source(sm120 / "instantiations" / "inst_q64_k64_d128_nvfp4.cu"),
                    _source(sm120 / "instantiations" / "inst_q128_k64_d128_nv_mx_fp16.cu"),
                    _source(sm120 / "instantiations" / "inst_q64_k64_d128_nv_mx_fp16.cu"),
                    _source(sm120 / "instantiations" / "inst_q128_k64_d128_nv_int8_fp16.cu"),
                    _source(sm120 / "instantiations" / "inst_q64_k64_d128_nv_int8_fp16.cu"),
                    _source(sm120 / "instantiations" / "inst_q64_k64_d128_mxfp8.cu"),
                    _source(sm120 / "q128_microscaling_preparation.cu"),
                ],
                include_dirs=[
                    str(attention),
                    str(attention.parent),
                    str(sm120),
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
