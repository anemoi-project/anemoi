#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="anemoi"
dry_run=0

if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
if (( "$#" != 0 )); then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found; install an NVIDIA driver first." >&2
  exit 1
fi

driver_version="$(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d' |
    sort -V |
    head -n 1 |
    tr -d '[:space:]'
)"
if [[ ! "${driver_version}" =~ ^[0-9]+([.][0-9]+)+$ ]]; then
  echo "could not determine the NVIDIA driver version" >&2
  exit 1
fi

version_at_least() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

if version_at_least "${driver_version}" "580.65.06"; then
  cuda_version="13.0"
  cuda_label="cuda-13.0.0"
  torch_version="2.11.0"
  torch_index="cu130"
elif version_at_least "${driver_version}" "570.26"; then
  cuda_version="12.8"
  cuda_label="cuda-12.8.0"
  torch_version="2.11.0"
  torch_index="cu128"
elif version_at_least "${driver_version}" "560.28.03"; then
  cuda_version="12.6"
  cuda_label="cuda-12.6.3"
  torch_version="2.7.1"
  torch_index="cu126"
else
  echo "NVIDIA driver ${driver_version} is too old for Anemoi's native SM89 path." >&2
  echo "Upgrade to at least 560.28.03 and retry." >&2
  exit 1
fi

echo "Detected NVIDIA driver ${driver_version}"
echo "Selected CUDA ${cuda_version}, PyTorch ${torch_version}+${torch_index}"
if (( dry_run == 1 )); then
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda or Anaconda and retry." >&2
  exit 1
fi

if conda run --no-capture-output -n "${env_name}" python --version >/dev/null 2>&1; then
  echo "Reusing existing conda environment: ${env_name}"
else
  echo "Creating conda environment: ${env_name} (Python 3.12)"
  conda create --yes --name "${env_name}" python=3.12 pip
fi

echo "Installing CUDA ${cuda_version} toolkit"
conda install --yes --name "${env_name}" \
  --channel "nvidia/label/${cuda_label}" cuda-toolkit

echo "Installing MiniMax-H3 runtime dependencies"
conda run --no-capture-output -n "${env_name}" \
  python -m pip install --upgrade pip
conda run --no-capture-output -n "${env_name}" \
  python -m pip install "torch==${torch_version}" \
  --index-url "https://download.pytorch.org/whl/${torch_index}"
conda run --no-capture-output -n "${env_name}" \
  python -m pip install -r "${repo_root}/requirements-minimax-h3.txt"

echo "Validating the selected CUDA stack"
conda run --no-capture-output -n "${env_name}" python - \
  "${cuda_version}" "${torch_version}" <<'PY'
import re
import subprocess
import sys

import torch


expected_cuda, expected_torch = sys.argv[1:]
nvcc = subprocess.run(
    [sys.prefix + "/bin/nvcc", "--version"],
    check=True,
    capture_output=True,
    text=True,
).stdout
match = re.search(r"release ([0-9]+[.][0-9]+)", nvcc)
observed_nvcc = match.group(1) if match else None
observed_torch = torch.__version__.split("+")[0]
if observed_nvcc != expected_cuda:
    raise SystemExit(f"nvcc {observed_nvcc!r} != selected CUDA {expected_cuda}")
if torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"PyTorch CUDA {torch.version.cuda!r} != selected CUDA {expected_cuda}"
    )
if observed_torch != expected_torch:
    raise SystemExit(f"PyTorch {observed_torch!r} != selected {expected_torch}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot initialize the NVIDIA driver")
print(
    f"Validated PyTorch {torch.__version__}, nvcc {observed_nvcc}, "
    f"GPU {torch.cuda.get_device_name(0)}"
)
PY

echo "Installing Anemoi and development tools"
cd "${repo_root}"
MPA_SKIP_CUDA_BUILD=1 conda run --no-capture-output -n "${env_name}" \
  python -m pip install --no-build-isolation --editable '.[dev]'

echo
echo "Anemoi environment is ready. Activate it with:"
echo "  conda activate ${env_name}"
echo "  export MPA_CUDA_HOME=\"\${CONDA_PREFIX}\""
