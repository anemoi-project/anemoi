#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
candidate="${1:-mpa-sm89-regular2d-mixed}"
output_dir="${2:-${repo_root}/outputs/minimax-h3/${candidate}}"
if [[ "$#" -ge 1 ]]; then shift; fi
if [[ "$#" -ge 1 ]]; then shift; fi

case "${candidate}" in
  dense|official-sol|mpa-sm89-regular2d-mixed) ;;
  *)
    echo "usage: $0 {dense|official-sol|mpa-sm89-regular2d-mixed} [OUTPUT_DIR] [runner args...]" >&2
    exit 2
    ;;
esac

resource_root="${EVG_MINIMAX_H3_ROOT:-${repo_root}/models/minimax-h3}"
export HF_HOME="${HF_HOME:-/dev/shm/evg-h3-hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/evg-h3-xdg}"
export TMPDIR="${TMPDIR:-/dev/shm}"
if [[ -n "${EVG_PYTHON:-}" ]]; then
  python_bin="${EVG_PYTHON}"
else
  venv_root="${EVG_MINIMAX_H3_VENV:-${resource_root}/.venv}"
  python_bin="${venv_root}/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    bootstrap_python="${EVG_BOOTSTRAP_PYTHON:-python3}"
    "${bootstrap_python}" -m venv "${venv_root}"
  fi
fi

if [[ "${EVG_INSTALL_DEPS:-1}" == "1" ]]; then
  if ! "${python_bin}" -c 'import accelerate, av, huggingface_hub, safetensors, torch, transformers, triton' >/dev/null 2>&1; then
    "${python_bin}" -m pip install --upgrade pip
    "${python_bin}" -m pip install -r "${repo_root}/requirements-minimax-h3.txt"
  fi
fi

if ! "${python_bin}" -c 'import huggingface_hub, torch, triton' >/dev/null 2>&1; then
  echo "MiniMax-H3 dependencies are unavailable; set EVG_INSTALL_DEPS=1 or provide EVG_PYTHON" >&2
  exit 1
fi

if [[ -n "${H3_DIFFUSERS_CHECKOUT:-}" || -n "${H3_MODEL_ROOT:-}" || -n "${H3_DIT_CHECKPOINT:-}" ]]; then
  : "${H3_DIFFUSERS_CHECKOUT:?set all three H3 resource variables or unset all of them}"
  : "${H3_MODEL_ROOT:?set all three H3 resource variables or unset all of them}"
  : "${H3_DIT_CHECKPOINT:?set all three H3 resource variables or unset all of them}"
else
  (
    cd "${repo_root}"
    "${python_bin}" -m evg.models.minimax_h3.resources --root "${resource_root}"
  )
  export H3_DIFFUSERS_CHECKOUT="${resource_root}/diffusers"
  export H3_MODEL_ROOT="${resource_root}/model"
  export H3_DIT_CHECKPOINT="${resource_root}/checkpoint/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
fi

export MPA_PYTHON="${python_bin}"

if [[ "${candidate}" == "mpa-sm89-regular2d-mixed" ]]; then
  "${repo_root}/scripts/build_router_cuda.sh"
  "${repo_root}/scripts/build_attention_cuda.sh"
fi

single_process=0
height_set=0
width_set=0
for argument in "$@"; do
  if [[ "${argument}" == "--decode-only" || "${argument}" == "--print-config" ]]; then
    single_process=1
  fi
  case "${argument}" in
    --height|--height=*) height_set=1 ;;
    --width|--width=*) width_set=1 ;;
  esac
done
if (( height_set != width_set )); then
  echo "set --height and --width together" >&2
  exit 2
fi

runner_args=(
  --candidate "${candidate}"
  --output-dir "${output_dir}"
  --diffusers-src "${H3_DIFFUSERS_CHECKOUT}"
  --model-root "${H3_MODEL_ROOT}"
  --checkpoint "${H3_DIT_CHECKPOINT}"
  "$@"
)

cd "${repo_root}"
if [[ "${single_process}" == "1" ]]; then
  exec "${python_bin}" -m evg.models.minimax_h3.runner "${runner_args[@]}"
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

if [[ -n "${EVG_NUM_GPUS:-}" ]]; then
  num_gpus="${EVG_NUM_GPUS}"
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
  echo "EVG_NUM_GPUS must be a positive integer, got ${num_gpus}" >&2
  exit 1
fi
if (( num_gpus < 1 || num_gpus > visible_gpus )); then
  echo "EVG_NUM_GPUS=${num_gpus} is invalid for ${visible_gpus} visible GPUs" >&2
  exit 1
fi
if (( 56 % num_gpus != 0 )); then
  echo "MiniMax-H3's 56 attention heads are not divisible by ${num_gpus} GPUs" >&2
  exit 1
fi

if (( height_set == 0 && num_gpus <= 2 )); then
  min_gpu_memory_mib="$(EVG_SELECTED_GPUS="${num_gpus}" "${python_bin}" -c '
import os
import torch
count = int(os.environ["EVG_SELECTED_GPUS"])
print(min(torch.cuda.get_device_properties(index).total_memory for index in range(count)) // (1 << 20))
')"
  if (( min_gpu_memory_mib < 32768 )); then
    runner_args+=(--height 480 --width 832)
    echo "Using 832x480 for ${num_gpus} GPU(s); minimum per-device memory is ${min_gpu_memory_mib} MiB."
  fi
fi

if [[ "${candidate}" == "mpa-sm89-regular2d-mixed" ]]; then
  EVG_SELECTED_GPUS="${num_gpus}" "${python_bin}" -c '
import os
import torch
count = int(os.environ["EVG_SELECTED_GPUS"])
capabilities = [torch.cuda.get_device_capability(index) for index in range(count)]
if any(capability != (8, 9) for capability in capabilities):
    raise SystemExit(f"MPA requires SM89 GPUs, got {capabilities}")
'
fi

exec "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${num_gpus}" \
  --module evg.models.minimax_h3.runner \
  "${runner_args[@]}"
