#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${MPA_PYTHON:-python}"

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 {dense|official-sol|mpa-sm89-regular2d-mixed} OUTPUT_DIR [runner args...]" >&2
  exit 2
fi

candidate="$1"
output_dir="$2"
shift 2

case "${candidate}" in
  dense|official-sol|mpa-sm89-regular2d-mixed) ;;
  *) echo "unsupported candidate: ${candidate}" >&2; exit 2 ;;
esac

: "${H3_DIFFUSERS_CHECKOUT:?set H3_DIFFUSERS_CHECKOUT to the pinned diffusers checkout}"
: "${H3_MODEL_ROOT:?set H3_MODEL_ROOT to the MiniMax-H3 Diffusers model root}"
: "${H3_DIT_CHECKPOINT:?set H3_DIT_CHECKPOINT to the pruned-FP8 checkpoint}"

export HF_HOME="${HF_HOME:-/dev/shm/evg-h3-hf}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/dev/shm/evg-h3-triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/evg-h3-inductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/evg-h3-xdg}"
export TMPDIR="${TMPDIR:-/dev/shm}"

exec "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  "${repo_root}/benchmarks/run_minimax_h3_fp8_ulysses_sm89.py" \
  --candidate "${candidate}" \
  --output-dir "${output_dir}" \
  --diffusers-src "${H3_DIFFUSERS_CHECKOUT}" \
  --model-root "${H3_MODEL_ROOT}" \
  --checkpoint "${H3_DIT_CHECKPOINT}" \
  "$@"
