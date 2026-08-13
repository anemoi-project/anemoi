# EVG Project

Efficient Visual Generation (EVG) is an inference-only framework for fast visual generation, especially for fast video generation. The current executable model family is MiniMax-H3 with EVG's adaptive [Draft Attention](https://arxiv.org/pdf/2505.14708) routing and native SM89 FP8/FP16 Mixed-Precision Attention (MPA).

## MiniMax-H3 quick start

The launcher creates an isolated environment when needed, downloads only the required model components and compressed DiT, builds the native MPA extensions, selects a valid number of visible GPUs, and generates the demo video:

```bash
scripts/run_minimax_h3.sh
```

By default, resources are stored under the ignored
`models/minimax-h3/` directory and output is written to
`outputs/minimax-h3/mpa-sm89-regular2d-mixed/out.mp4`.

Requirements:

- Linux and CUDA with `nvcc`
- one or more NVIDIA GPUs
- Python 3.12 with `venv` support
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

See [the reproduction guide](evg/models/minimax_h3/REPRODUCTION.md) for pinned resource revisions, manual setup, smoke tests, output contracts, and resource overrides.

## Repository layout

- `evg/models/minimax_h3/`: model adapter, runner, resource downloader, and
  package-local Sol runtime
- `evg/layers/attention/mpa/`: reusable MPA routing and SM89 backend
- `csrc/`: native router and mixed-attention CUDA sources
- `scripts/run_minimax_h3.sh`: download-to-video entry point
- `scripts/build_*_cuda.sh`: standalone native-extension builds

The remaining model registry and generic Draft Attention implementation are kept as framework infrastructure; MiniMax-H3 is the current validated end-to-end path.

## Development checks

```bash
python -m pip install -e .[dev]
python -m unittest discover -s tests
```

## Organization
- Zhejiang University
- Nanjing University