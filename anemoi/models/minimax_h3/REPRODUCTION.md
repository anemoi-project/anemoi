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
MPA_BUILD_COMPONENTS=sm120 \
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

The launcher applies one architecture-neutral Q64 pure-INT8 policy on SM89 and
SM120. Runtime device capability selects the native backend. The canonical
configuration is
[`examples/minimax-h3/mpa-ragged2d-mixed.yaml`](../../../examples/minimax-h3/mpa-ragged2d-mixed.yaml).
Reference it explicitly with:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa \
  --mpa-config examples/minimax-h3/mpa-ragged2d-mixed.yaml
```

The SM120-named file is a compatibility alias with the same parsed policy. For
example:

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

The stable YAML controls query geometry, base and per-layer sparsity,
retained-block precision, optional SM89/SM120 Mean/MaxPool routing fusion, and
dense-first steps/layers. It uses the same field set as `SparseConfig` and
`QuantConfig`, plus the H3 adapter's dense schedule. Unknown fields and invalid
ranges are rejected.

The fields have the following meanings:

| Field | Meaning |
| --- | --- |
| `query_block_size` | Logical query block size, either `64` or `128`. Both sizes are supported by the SM89 and SM120 native paths. |
| `prefix_kv_precision`, `prefix_query_precision` | Prefix K/V and prefix-query arithmetic precision. The production value is `int8` on both SM89 and SM120. Unsupported architecture/precision combinations are rejected before kernel launch. |
| `sparsity_ratio` | Fraction of routed video block pairs dropped outside `layer_sparsity_bands`. The frozen default is `0.80` for every sparse layer. |
| `layer_sparsity_bands` | Zero-based, half-open per-layer overrides. The canonical file inherits the empty default `()` from `SparseConfig`, so no layer changes the global 80% sparsity unless bands are supplied explicitly. |
| `maxpool_weight` | SM89/SM120 Q64/Q128 Mean/MaxPool routing blend in `[0, 1]`. The default `0` preserves mean-only routing; `1` is max-only; interior values fuse independently normalized mean and max probability maps. |
| `nvfp4_ratio`, `int8_ratio`, `fp16_ratio` | Stable retained-block precision fractions. They must be finite, nonnegative, and sum to one. The production value is pure INT8. SM89 accepts only INT8/FP16; NVFP4 requires SM120. |
| `dense_first_steps` | Number of initial denoising steps that use original-dtype dense SDPA for every transformer layer. The released value is 10. |
| `dense_first_layers` | Number of leading transformer layers that remain dense at every denoising step. With value 2, zero-based layers 0 and 1 remain dense after the initial dense-only steps. |

The dense guards take precedence over sparsity settings: a call selected by
`dense_first_steps` or `dense_first_layers` does not enter the sparse INT8
path. The resolved policy, including the derived average sparse-layer ratio, is
available through `--print-config` and is recorded with the run artifacts.

For a precision-only ablation, command-line overrides are also available and
take precedence over the YAML file.

The production default on both architectures is Q64 pure INT8. A legacy SM89
mixed-phase CLI ablation must provide two positive ratios that sum to one:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa-70-30 \
  --fp8-ratio 0.7 --fp16-ratio 0.3
```

Do not report an overridden split as the released default.

On SM89, both the portable `int8_ratio` and legacy routed-video `fp8_ratio`
name the established low arithmetic: Q/K use symmetric INT8 tensor-core
operands and V uses E4M3. Prefix K/V reuses those already prepared K/V tensors,
while prefix queries use direct strided Q64 quantization and a dense-sequential
specialization of the same production phase body. Pure INT8 dispatches the
existing no-FP16 specialization, and no dense block-ID matrix is allocated.

Historical ablations use a separate private experimental schema. The release
runner accepts only the stable fields documented above.
