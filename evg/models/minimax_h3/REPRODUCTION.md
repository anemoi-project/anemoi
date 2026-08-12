# MiniMax-H3 reproduction

This guide reproduces the Dense, official Sol, and EVG MPA candidates with a
single controlled MiniMax-H3 inference stack. Dense is the quality baseline.

## Automated path

From the EVG repository root:

```bash
scripts/run_minimax_h3.sh \
  mpa-sm89-regular2d-mixed \
  outputs/minimax-h3/mpa
```

On the first run, the script:

1. creates `models/minimax-h3/.venv` unless `EVG_PYTHON` is supplied;
2. installs `requirements-minimax-h3.txt` if imports are unavailable;
3. downloads the pinned Diffusers source and required model files;
4. verifies the compressed checkpoint size and SHA256;
5. builds the SM89 router and attention extensions for MPA;
6. chooses a valid Ulysses degree and runs denoise plus decode.

Set `EVG_NUM_GPUS` to control the process count. MiniMax-H3 has 56 attention
heads, so the value must divide 56 and cannot exceed the visible GPU count. The
automatic selector considers 8, 7, 4, 2, and 1 GPUs in that order. The MPA
candidate additionally requires every selected GPU to have SM89 capability.
When only one or two GPUs with less than 32 GiB each are selected and no
resolution is specified, the launcher uses the 832x480 memory-safe demo
profile. Four 24 GiB GPUs and larger-memory configurations retain 1344x768.
Pass both `--height` and `--width` to choose a resolution explicitly.

## Pinned external resources

- Diffusers repository: `https://github.com/huggingface/diffusers.git`
- Diffusers commit: `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`
- model repository: `MiniMaxAI/MiniMax-H3`
- model revision: `939557dc319dd91227e30195a763f272ba7f8765`
- compressed DiT repository: `Comfy-Org/MiniMax-H3`
- compressed DiT revision: `014cd40f7e177756c6b2473c0d93b1c89a790dd2`
- compressed DiT file:
  `diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors`

The compressed checkpoint contract is:

```text
size:   20958205608 bytes
sha256: 12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993
```

Only `transformer/config.json`, schedulers, VAE, audio VAE, and the modular
model index are downloaded from the full MiniMax-H3 repository. The BF16
transformer and text encoder are not on this run path. Prompt conditioning is
bundled at `evg/models/minimax_h3/assets/official-example-1.pt`.

Use `HF_TOKEN` if the Hugging Face client requires authentication. Change the
managed download directory with `EVG_MINIMAX_H3_ROOT`.

## Manual environment and download

The validated setup uses Python 3.12, CUDA toolkit 12.9.1, Torch 2.11.0, and
Triton 3.6.0. To prepare it manually:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-minimax-h3.txt

python -m evg.models.minimax_h3.resources \
  --root models/minimax-h3
```

You may bypass managed downloads by setting all three paths together:

```bash
export H3_DIFFUSERS_CHECKOUT=/absolute/path/to/diffusers
export H3_MODEL_ROOT=/absolute/path/to/MiniMax-H3-diffusers
export H3_DIT_CHECKPOINT=/absolute/path/to/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
export EVG_PYTHON="$PWD/.venv/bin/python"
```

The launcher refuses a partial override so files from different resource sets
cannot be mixed accidentally.

## Manual native build

```bash
export MPA_PYTHON="$PWD/.venv/bin/python"
export MPA_CUDA_HOME=/absolute/path/to/cuda-12.9
export MPA_BUILD_ROOT=/dev/shm/evg-mpa-build

scripts/build_router_cuda.sh
scripts/build_attention_cuda.sh
```

The extensions are installed in `evg.layers.attention.mpa` for the active
Python ABI. Temporary objects and runtime caches default to `/dev/shm`.

## Fast smoke test

Use twelve scheduler points and skip decode to validate model load,
distributed exchange, the ten-step dense-first interval, and the selected
sparse attention path:

```bash
EVG_NUM_GPUS=2 scripts/run_minimax_h3.sh \
  mpa-sm89-regular2d-mixed \
  /dev/shm/evg-h3-smoke \
  --steps 12 --no-decode
```

Choose a GPU count available on the machine. A successful output contains
`benchmark.json`, `run_config.json`, and `denoised_state.pt`. Require
`rank_outputs_bit_identical=true`, one `per_rank` entry per process, and a
positive `attention_stats.mpa_calls` value. A two-step run is a faster
load-and-memory check, but it stays entirely inside the dense-first interval.

## Controlled 50-step comparison

Keep every runner argument identical:

```bash
for candidate in dense official-sol mpa-sm89-regular2d-mixed; do
  scripts/run_minimax_h3.sh \
    "$candidate" \
    "outputs/minimax-h3/$candidate" \
    --height 768 --width 1344
done
```

This established quality workload uses seed 0, 50 scheduler steps, 124
generated frames, 1344x768 resolution, and the bundled official conditioning.
It requires sufficient GPU memory. On a smaller machine, omit the explicit
dimensions so every candidate uses the same automatic 832x480 profile. The
final video contains 120 frames at 24 FPS with audio.

Each output directory contains:

- `out.mp4`: canonical video;
- `<candidate>_<width>x<height>_5s_24fps.mp4`: named video;
- `denoised_state.pt`: CPU latent artifact;
- `benchmark.json`: timing, memory, policy, hashes, and media probe;
- `run_config.json`: resolved workload and distributed configuration.

## Decode an existing latent

The launcher detects `--decode-only` and runs it in one process:

```bash
scripts/run_minimax_h3.sh \
  mpa-sm89-regular2d-mixed \
  /absolute/path/to/existing/output \
  --decode-only
```

## Optional precision split

The released MPA default is FP8/FP16 = 0.8/0.2. An ablation must provide both
positive ratios and they must sum to one:

```bash
scripts/run_minimax_h3.sh \
  mpa-sm89-regular2d-mixed \
  outputs/minimax-h3/mpa-70-30 \
  --fp8-ratio 0.7 --fp16-ratio 0.3
```

Do not report an overridden split as the released default.
