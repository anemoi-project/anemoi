#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${MPA_PYTHON:-python}"
cuda_root="${MPA_CUDA_HOME:-${CUDA_HOME:-}}"
if [[ -z "${cuda_root}" ]]; then
  cuda_root="$("${python_bin}" -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or "")')"
fi
if [[ -z "${cuda_root}" || ! -x "${cuda_root}/bin/nvcc" ]]; then
  echo "set MPA_CUDA_HOME to a CUDA toolkit containing bin/nvcc" >&2
  exit 1
fi

export CUDA_HOME="${cuda_root}"
export PATH="${cuda_root}/bin:${PATH}"
components="${MPA_BUILD_COMPONENTS:-sm89}"
default_arch="8.9"
if [[ ",${components}," == *",sm120_q64,"* ]]; then
  default_arch="12.0a"
fi
export TORCH_CUDA_ARCH_LIST="${MPA_TORCH_CUDA_ARCH_LIST:-${default_arch}}"
export MAX_JOBS="${MPA_MAX_JOBS:-4}"

build_root="${MPA_BUILD_ROOT:-${TMPDIR:-/tmp}/anemoi-mpa-build-${UID}}"
mkdir -p "${build_root}/attention/temp" "${build_root}/attention/lib"
cd "${repo_root}"
MPA_BUILD_COMPONENTS="${components}" "${python_bin}" setup.py build_ext \
  --build-temp "${build_root}/attention/temp" \
  --build-lib "${build_root}/attention/lib" \
  --inplace
