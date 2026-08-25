# MiniMax-H3 reproduction

This guide reproduces the Dense, official Sol, and Anemoi MPA candidates with a
single controlled MiniMax-H3 inference stack. Dense is the quality baseline.

## Automated path

From the Anemoi repository root:

```bash
scripts/setup_conda_env.sh
conda activate anemoi

scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa
```

The environment setup is required only once. On the first model run, the
launcher:

1. uses the active `anemoi` Conda environment (or `ANEMOI_PYTHON` when supplied);
2. downloads the pinned Diffusers source and required model files;
3. verifies the compressed checkpoint size and SHA256;
4. builds the native attention extension for the selected architecture;
5. chooses a valid Ulysses degree and runs denoise plus decode.

The setup script selects a stack from the lowest driver version across visible
GPUs:

- driver 580.65.06 or newer: CUDA 13.0 and Torch 2.11 cu130;
- driver 570.26 or newer: CUDA 12.8 and Torch 2.11 cu128;
- driver 560.28.03 or newer: CUDA 12.6 and Torch 2.7 cu126.

Run `scripts/setup_conda_env.sh --dry-run` to inspect the selection without
creating or updating the environment. Older drivers are rejected because they
are outside the supported native build profiles.

Set `ANEMOI_NUM_GPUS` to control the process count. MiniMax-H3 has 56 attention
heads, so the value must divide 56 and cannot exceed the visible GPU count. The
automatic selector considers 8, 7, 4, 2, and 1 GPUs in that order. The MPA
candidate additionally requires a homogeneous set of SM89 or SM120 GPUs.
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
bundled at `anemoi/models/minimax_h3/assets/official-example-1.pt`.

Use `HF_TOKEN` if the Hugging Face client requires authentication. Change the
managed download directory with `ANEMOI_MINIMAX_H3_ROOT`.

## Manual environment and download

The setup script installs Python 3.12 and a driver-compatible CUDA/Torch
profile. If it cannot be used, prepare a virtual environment manually and
install one of the profiles above before the remaining requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Example for the CUDA 12.6 profile:
python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-minimax-h3.txt

python -m anemoi.models.minimax_h3.resources \
  --root models/minimax-h3
```

You may bypass managed downloads by setting all three paths together:

```bash
export H3_DIFFUSERS_CHECKOUT=/absolute/path/to/diffusers
export H3_MODEL_ROOT=/absolute/path/to/MiniMax-H3-diffusers
export H3_DIT_CHECKPOINT=/absolute/path/to/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
export ANEMOI_PYTHON="$PWD/.venv/bin/python"
```

The launcher refuses a partial override so files from different resource sets
cannot be mixed accidentally.

## Manual native build

```bash
export MPA_PYTHON="$PWD/.venv/bin/python"
export MPA_CUDA_HOME=/absolute/path/to/selected-conda-environment
export MPA_BUILD_ROOT="${TMPDIR:-/tmp}/anemoi-mpa-build-${UID}"

scripts/build_attention_cuda.sh
```

The extensions are installed in `anemoi.layers.attention.mpa` for the active
Python ABI. Temporary objects and runtime caches default to `/dev/shm`.

For an SM120 build, select the native component and architecture explicitly:

```bash
MPA_BUILD_COMPONENTS=sm89,sm120_q64 \
MPA_TORCH_CUDA_ARCH_LIST=12.0a \
scripts/build_attention_cuda.sh
```

## Fast smoke test

Use twelve scheduler points and skip decode to validate model load,
distributed exchange, the ten-step dense-first interval, and the selected
sparse attention path:

```bash
ANEMOI_NUM_GPUS=2 scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  /dev/shm/anemoi-h3-smoke \
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
for candidate in dense official-sol mpa-ragged2d-mixed; do
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
  mpa-ragged2d-mixed \
  /absolute/path/to/existing/output \
  --decode-only
```

## MPA configuration

The launcher selects the released Q64 policy by GPU architecture: SM89 uses
[`examples/minimax-h3/mpa-ragged2d-mixed.yaml`](../../../examples/minimax-h3/mpa-ragged2d-mixed.yaml),
while SM120 uses pure INT8 from
[`examples/minimax-h3/mpa-sm120-q64-int8.yaml`](../../../examples/minimax-h3/mpa-sm120-q64-int8.yaml).
Reference the SM89 policy explicitly with:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa \
  --mpa-config examples/minimax-h3/mpa-ragged2d-mixed.yaml
```

The SM120 pure-INT8 default can be stated explicitly without adding another
candidate name:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa-sm120-q64 \
  --mpa-config examples/minimax-h3/mpa-sm120-q64-int8.yaml
```

Copy the file before changing an experiment:

```bash
cp examples/minimax-h3/mpa-ragged2d-mixed.yaml \
  /tmp/my-mpa-config.yaml

scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa-custom \
  --mpa-config /tmp/my-mpa-config.yaml
```

The YAML controls query geometry, base and per-layer sparsity, retained-block
precision phases, route scoring, and dense-first steps/layers. Unknown fields
and invalid ranges are rejected instead of being silently ignored.

The fields have the following meanings:

| Field | Meaning |
| --- | --- |
| `query_block_size` | Logical query block size, either `64` or `128`. Q128 selects the SM120 path and is rejected on SM89. |
| `prefix_kv_precision`, `prefix_query_precision` | Prefix K/V and prefix-query arithmetic precision: `auto`, `fp16`, `mxfp8`, `nvfp4`, or `int8`. |
| `sparsity_ratio` | Fraction of routed video block pairs dropped in MPA layers that are not covered by `layer_sparsity_bands`. The released value `0.88` retains approximately 12% of the block pairs. The router rounds the retained count and always keeps at least one pair. Missing mandatory anchors replace the weakest ordinary edges in the lowest configured precision, so anchors do not expand the retained budget. |
| `layer_sparsity_bands` | Per-layer overrides written as `[first, last, sparsity]`. Layer ranges are zero-based and half-open: `[18, 34, 0.82]` applies 82% sparsity to layers 18 through 33. Bands must be sorted, non-overlapping, and within the 50-layer transformer stack. Layers outside the bands use `sparsity_ratio`. |
| `fp8_ratio` | SM89 target fraction of retained sparse block pairs assigned to FP8. |
| `nvfp4_ratio`, `int8_ratio`, `mxfp8_ratio`, `fp16_ratio` | SM120 retained-block precision fractions. They must be finite, nonnegative, and sum to one; INT8 and MXFP8 are alternative middle phases. On SM89, `fp16_ratio` instead completes the positive FP8/FP16 pair. |
| `layer_precision_bands` | Optional SM120 per-layer overrides written as `[first, last, NVFP4, INT8, [MXFP8,] FP16]`; each band must sum to one. |
| `enable_anchors` | Keep mandatory same-frame adjacency anchors. An anchor already selected by DraftMap keeps its precision; a missing anchor replaces the weakest ordinary edge in the lowest configured precision budget. Configurations whose lowest-precision budget is smaller than the static anchor set are rejected. The SM89 and SM120 Q64 production presets enable anchors. |
| `diag_jensen` | Boolean switch for the optional diagonal-Jensen second-moment correction. `false` uses the fixed pooled-QK row-softmax probability score; `true` adds the correction. |
| `dense_first_steps` | Number of initial denoising steps that use original-dtype dense SDPA for every transformer layer. The released value is 10. |
| `dense_first_layers` | Number of leading transformer layers that remain dense at every denoising step. With value 2, zero-based layers 0 and 1 remain dense after the initial dense-only steps. |

The dense guards take precedence over sparsity settings: a call selected by
`dense_first_steps` or `dense_first_layers` does not enter the sparse FP8/FP16
path. The resolved policy, including the derived average sparse-layer ratio, is
available through `--print-config` and is recorded with the run artifacts.

For a precision-only ablation, command-line overrides are also available and
take precedence over the YAML file.

The released SM89 default is FP8/FP16 = 0.8/0.2; the released SM120 default is
Q64 pure INT8. An SM89 ablation must provide both positive ratios and they must
sum to one:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa-70-30 \
  --fp8-ratio 0.7 --fp16-ratio 0.3
```

Do not report an overridden split as the released default.
