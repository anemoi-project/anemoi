#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
candidate="${1:-mpa-ragged2d-mixed}"
output_dir="${2:-${repo_root}/outputs/minimax-h3/${candidate}}"
if [[ "$#" -ge 1 ]]; then shift; fi
if [[ "$#" -ge 1 ]]; then shift; fi

case "${candidate}" in
  dense|official-sol|mpa-ragged2d-mixed) ;;
  *)
    echo "usage: $0 {dense|official-sol|mpa-ragged2d-mixed} [OUTPUT_DIR] [runner args...]" >&2
    exit 2
    ;;
esac

resource_root="${ANEMOI_MINIMAX_H3_ROOT:-${repo_root}/models/minimax-h3}"
export TMPDIR="${TMPDIR:-/dev/shm}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
cache_namespace="anemoi-h3-$(id -u)"
export HF_HOME="${HF_HOME:-${TMPDIR}/${cache_namespace}/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR}/${cache_namespace}/xdg}"
if [[ -n "${ANEMOI_PYTHON:-}" ]]; then
  python_bin="${ANEMOI_PYTHON}"
  python_path="$(command -v "${python_bin}" 2>/dev/null || true)"
  if [[ -n "${python_path}" ]]; then
    python_prefix="$(cd "$(dirname "${python_path}")/.." && pwd)"
  fi
  if [[ -n "${python_prefix:-}" && -x "${python_prefix}/bin/nvcc" ]]; then
    export MPA_CUDA_HOME="${MPA_CUDA_HOME:-${python_prefix}}"
  fi
elif [[ "${CONDA_DEFAULT_ENV:-}" == "anemoi" && -n "${CONDA_PREFIX:-}" ]]; then
  python_bin="${CONDA_PREFIX}/bin/python"
  if [[ -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export MPA_CUDA_HOME="${MPA_CUDA_HOME:-${CONDA_PREFIX}}"
  fi
else
  venv_root="${ANEMOI_MINIMAX_H3_VENV:-${resource_root}/.venv}"
  python_bin="${venv_root}/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    bootstrap_python="${ANEMOI_BOOTSTRAP_PYTHON:-python3}"
    "${bootstrap_python}" -m venv "${venv_root}"
  fi
fi

if [[ "${ANEMOI_INSTALL_DEPS:-1}" == "1" ]]; then
  if ! "${python_bin}" -c 'import accelerate, av, huggingface_hub, safetensors, torch, transformers, triton, yaml' >/dev/null 2>&1; then
    "${python_bin}" -m pip install --upgrade pip
    "${python_bin}" -m pip install -r "${repo_root}/requirements-minimax-h3.txt"
  fi
fi

if ! "${python_bin}" -c 'import huggingface_hub, torch, triton, yaml' >/dev/null 2>&1; then
  echo "MiniMax-H3 dependencies are unavailable; set ANEMOI_INSTALL_DEPS=1 or provide ANEMOI_PYTHON" >&2
  exit 1
fi
if [[ -n "${H3_DIFFUSERS_CHECKOUT:-}" || -n "${H3_MODEL_ROOT:-}" || -n "${H3_DIT_CHECKPOINT:-}" ]]; then
  : "${H3_DIFFUSERS_CHECKOUT:?set all three H3 resource variables or unset all of them}"
  : "${H3_MODEL_ROOT:?set all three H3 resource variables or unset all of them}"
  : "${H3_DIT_CHECKPOINT:?set all three H3 resource variables or unset all of them}"
else
  (
    cd "${repo_root}"
    "${python_bin}" -m anemoi.models.minimax_h3.resources --root "${resource_root}"
  )
  export H3_DIFFUSERS_CHECKOUT="${resource_root}/diffusers"
  export H3_MODEL_ROOT="${resource_root}/model"
  export H3_DIT_CHECKPOINT="${resource_root}/checkpoint/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
fi

export MPA_PYTHON="${python_bin}"

single_process=0
height_set=0
width_set=0
mpa_config_set=0
for argument in "$@"; do
  if [[ "${argument}" == "--decode-only" || "${argument}" == "--print-config" ]]; then
    single_process=1
  fi
  case "${argument}" in
    --height|--height=*) height_set=1 ;;
    --width|--width=*) width_set=1 ;;
    --mpa-config|--mpa-config=*) mpa_config_set=1 ;;
  esac
done
if (( height_set != width_set )); then
  echo "set --height and --width together" >&2
  exit 2
fi

sm120_selected=0
if [[ "${candidate}" == "mpa-ragged2d-mixed" && "${single_process}" == "0" ]]; then
  if "${python_bin}" -c 'import torch; raise SystemExit(torch.cuda.get_device_capability(0) != (12, 0))'; then
    sm120_selected=1
    export MPA_BUILD_COMPONENTS="${MPA_BUILD_COMPONENTS:-sm89,sm120_q64}"
    build_revision="$(git -C "${repo_root}" rev-parse --short=12 HEAD 2>/dev/null || printf source)"
    export MPA_BUILD_ROOT="${MPA_BUILD_ROOT:-${TMPDIR}/anemoi-mpa-build-sm120-${UID}-${build_revision}}"
  fi
  "${repo_root}/scripts/build_attention_cuda.sh"
fi

runner_args=(
  --candidate "${candidate}"
  --output-dir "${output_dir}"
  --diffusers-src "${H3_DIFFUSERS_CHECKOUT}"
  --model-root "${H3_MODEL_ROOT}"
  --checkpoint "${H3_DIT_CHECKPOINT}"
  "$@"
)
if (( sm120_selected == 1 && mpa_config_set == 0 )); then
  runner_args+=(
    --mpa-config "${repo_root}/examples/minimax-h3/mpa-sm120-q64-int8.yaml"
  )
fi

cd "${repo_root}"
if [[ "${single_process}" == "1" ]]; then
  exec "${python_bin}" -m anemoi.models.minimax_h3.runner "${runner_args[@]}"
fi

visible_gpus="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ ! "${visible_gpus}" =~ ^[0-9]+$ ]]; then
  echo "PyTorch returned an invalid GPU count: ${visible_gpus}" >&2
  exit 1
fi
if (( visible_gpus < 1 )); then
  echo "no CUDA GPU is visible to PyTorch" >&2
  exit 1
fi

if [[ -n "${ANEMOI_NUM_GPUS:-}" ]]; then
  num_gpus="${ANEMOI_NUM_GPUS}"
else
  num_gpus=0
  for degree in 8 7 4 2 1; do
    if [[ "${degree}" -le "${visible_gpus}" ]]; then
      num_gpus="${degree}"
      break
    fi
  done
fi
if [[ ! "${num_gpus}" =~ ^[0-9]+$ ]]; then
  echo "ANEMOI_NUM_GPUS must be a positive integer, got ${num_gpus}" >&2
  exit 1
fi
if (( num_gpus < 1 || num_gpus > visible_gpus )); then
  echo "ANEMOI_NUM_GPUS=${num_gpus} is invalid for ${visible_gpus} visible GPUs" >&2
  exit 1
fi
if (( 56 % num_gpus != 0 )); then
  echo "MiniMax-H3's 56 attention heads are not divisible by ${num_gpus} GPUs" >&2
  exit 1
fi

if (( height_set == 0 && num_gpus <= 2 )); then
  min_gpu_memory_mib="$(ANEMOI_SELECTED_GPUS="${num_gpus}" "${python_bin}" -c '
import os
import torch
count = int(os.environ["ANEMOI_SELECTED_GPUS"])
print(min(torch.cuda.get_device_properties(index).total_memory for index in range(count)) // (1 << 20))
')"
  if (( min_gpu_memory_mib < 32768 )); then
    runner_args+=(--height 480 --width 832)
    echo "Using 832x480 for ${num_gpus} GPU(s); minimum per-device memory is ${min_gpu_memory_mib} MiB."
  fi
fi

if [[ "${candidate}" == "mpa-ragged2d-mixed" ]]; then
  ANEMOI_SELECTED_GPUS="${num_gpus}" "${python_bin}" -c '
import os
import torch
count = int(os.environ["ANEMOI_SELECTED_GPUS"])
capabilities = [torch.cuda.get_device_capability(index) for index in range(count)]
if any(capability not in ((8, 9), (12, 0)) for capability in capabilities):
    raise SystemExit(f"MPA requires SM89 or SM120 GPUs, got {capabilities}")
if len(set(capabilities)) != 1:
    raise SystemExit(f"MPA requires homogeneous GPU capabilities, got {capabilities}")
'
fi

exec "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${num_gpus}" \
  --module anemoi.models.minimax_h3.runner \
  "${runner_args[@]}"
