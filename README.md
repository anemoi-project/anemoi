# EVG Project

Efficient Visual Generation (EVG) is an inference-only framework for fast visual generation, especially for fast video generation. The current executable model family is MiniMax-H3 with EVG's adaptive [Draft Attention](https://arxiv.org/pdf/2505.14708) routing and native SM89 FP8/FP16 Mixed-Precision Attention (MPA).

## Installation

Linux, an NVIDIA GPU, and a CUDA toolkit containing `nvcc` are required for
the native MiniMax-H3 MPA path. We recommend Conda for environment management.
The setup script creates (or updates) an environment named `evg`, installs the
pinned MiniMax-H3 runtime dependencies, EVG itself, and the development tools:

```bash
scripts/setup_conda_env.sh
conda activate evg
```

It detects the lowest NVIDIA driver version across the visible GPUs and picks
a matching CUDA/PyTorch stack. Preview the selection without changing the
environment with `scripts/setup_conda_env.sh --dry-run`.

The script installs Python 3.12 and does not compile the CUDA extensions during
installation. The MiniMax-H3 launcher builds them for the active PyTorch and
CUDA installation when the MPA candidate is selected.

For framework-only development, run the checks after activating the environment:

```bash
python -m unittest discover -s tests
```

## MiniMax-H3 quick start

After activating the `evg` environment, the launcher downloads only the required
model components and compressed DiT, builds the native MPA extensions, selects
a valid number of visible GPUs, and generates the demo video:

```bash
scripts/run_minimax_h3.sh
```

By default, resources are stored under the ignored
`models/minimax-h3/` directory and output is written to
`outputs/minimax-h3/mpa-sm89-regular2d-mixed/out.mp4`.

Requirements:

- Linux and CUDA with `nvcc`
- one or more NVIDIA GPUs
- Conda (recommended), or Python 3.12 with `venv` support
- network access to GitHub and Hugging Face on the first run

The native MPA candidate requires SM89 GPUs such as RTX 4090. The launcher is not fixed to four GPUs: it automatically chooses the largest supported Ulysses degree from the visible devices. On one or two GPUs with less than 32 GiB each, it also selects the 832x480 demo profile to avoid running out of memory; four 24 GiB GPUs keep the validated 1344x768 profile. Explicit `--height` and `--width` values override this choice. Override the process count when needed:

```bash
EVG_NUM_GPUS=2 scripts/run_minimax_h3.sh
```

MiniMax-H3 has 56 attention heads, so the selected GPU count must divide 56. Memory headroom and performance still depend on the selected hardware.

Use the same runner for the controlled attention references:

```bash
scripts/run_minimax_h3.sh dense outputs/minimax-h3/dense
scripts/run_minimax_h3.sh official-sol outputs/minimax-h3/official-sol
scripts/run_minimax_h3.sh mpa-sm89-regular2d-mixed outputs/minimax-h3/mpa
```

Dense is the quality baseline; the state-of-the-art method [Sol-Attn](https://github.com/NVlabs/Sana/tree/sol-engine) is introduced as a reference implementation. All three paths use the same checkpoint, conditioning, model fusions, Ulysses degree, seed, scheduler, and decoder.

The released MPA policy is stored in
[`evg/models/minimax_h3/configs/mpa-sm89-regular2d-mixed.json`](evg/models/minimax_h3/configs/mpa-sm89-regular2d-mixed.json).
Copy it and pass `--mpa-config PATH` to change sparsity, the FP8/FP16 split,
tiling, or the dense-first schedule without editing Python source.

See [the reproduction guide](evg/models/minimax_h3/REPRODUCTION.md) for pinned resource revisions, manual setup, smoke tests, output contracts, and resource overrides.

## Repository layout

- `evg/models/minimax_h3/`: model adapter, runner, resource downloader, and
  package-local Sol runtime
- `evg/layers/attention/mpa/`: reusable MPA routing and SM89 backend
- `csrc/`: native router and mixed-attention CUDA sources
- `scripts/run_minimax_h3.sh`: download-to-video entry point
- `scripts/build_*_cuda.sh`: standalone native-extension builds

The remaining model registry and generic Draft Attention implementation are kept as framework infrastructure; MiniMax-H3 is the current validated end-to-end path.

## Organization
- Zhejiang University
- Nanjing University