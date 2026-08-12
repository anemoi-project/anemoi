# MiniMax-H3 SM89 demo reproduction

This guide reproduces the three attention candidates shipped with this
integration on four RTX 4090 GPUs. All three candidates use the same compressed
DiT, conditioning cache, model fusions, Ulysses degree, scheduler, seed, and
decode path. Only the post-Ulysses attention implementation changes.

## 1. Validated setup

- Linux with four NVIDIA RTX 4090 GPUs (SM89)
- Python 3.12
- CUDA toolkit 12.9.1 with `nvcc`
- the Python packages pinned in `requirements-demo.txt`
- enough host memory and local storage for the MiniMax-H3 resources

The runner requires exactly four visible GPUs. Compilation and framework caches
default to `/dev/shm`; keep them there to avoid slow builds or kernel timing
caused by a mounted filesystem. Result directories may be placed on persistent
storage.

Create an isolated environment, then install the pinned runtime:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
```

If the validated PyTorch wheel is not available from the default package index,
install the CUDA-compatible `torch==2.11.0` wheel first, then install the
remaining requirements without changing the Torch version.

## 2. External resources

The repository intentionally does not duplicate the model or the Diffusers
source tree. Prepare these three resources before building:

1. A Diffusers checkout at commit
   `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` from
   `https://github.com/huggingface/diffusers.git`. This is the MiniMax-H3
   implementation associated with Diffusers PR 14355.
2. A Diffusers-layout MiniMax-H3 model directory. The run path needs
   `transformer/config.json`, the scheduler/configuration files, and the VAE and
   audio-VAE weights. It does not load the BF16 transformer weights.
3. `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` from
   `Comfy-Org/MiniMax-H3`, under `diffusion_models/` in that repository.

The compressed DiT checkpoint is checked before every distributed run:

```text
size:   20958205608 bytes
sha256: 12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993
```

Export absolute paths to the resources:

```bash
export H3_DIFFUSERS_CHECKOUT=/absolute/path/to/diffusers
export H3_MODEL_ROOT=/absolute/path/to/MiniMax-H3-diffusers
export H3_DIT_CHECKPOINT=/absolute/path/to/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
```

Verify the source revision and checkpoint before continuing:

```bash
git -C "$H3_DIFFUSERS_CHECKOUT" rev-parse HEAD
stat -c '%s' "$H3_DIT_CHECKPOINT"
sha256sum "$H3_DIT_CHECKPOINT"
test -f "$H3_MODEL_ROOT/transformer/config.json"
test -f "$H3_MODEL_ROOT/vae/config.json"
test -f "$H3_MODEL_ROOT/audio_vae/config.json"
```

The official prompt conditioning is already included at
`assets/conditioning/official-example-1.pt`. Its file SHA256 is
`729514f5a3d74c8a61302ca39ace2d4db7fa065063bdf2b07cd003493f9bfd0c`.
The runner also validates the prompt identity stored inside the cache.

## 3. Build the SM89 extensions

Set the interpreter or CUDA toolkit explicitly when they are not the shell
defaults:

```bash
export MPA_PYTHON="$PWD/.venv/bin/python"
export MPA_CUDA_HOME=/absolute/path/to/cuda-12.9
export MPA_BUILD_ROOT=/dev/shm/evg-mpa-build

scripts/build_router_cuda.sh
scripts/build_attention_cuda.sh
```

Both scripts compile only for SM89 by default and place temporary objects under
`/dev/shm`. Successful builds create the two package-private extensions
`mpa._cuda_router` and `mpa._cuda_attention` in the working tree.

Check the resolved MPA policy without initializing CUDA or loading the model:

```bash
"$MPA_PYTHON" benchmarks/run_minimax_h3_fp8_ulysses_sm89.py \
  --candidate mpa-sm89-regular2d-mixed \
  --output-dir /dev/shm/config-only \
  --print-config
```

The released policy uses regular adaptive 2-D pooling (`24x42 -> 8x7` and
`24x40 -> 8x8`), physical Q64xK64 attention, probability-global routing with
legal same-frame 2-D cross anchors, the rising per-layer sparsity budget,
FP8/FP16 retained-block ratio 80/20, and no skip compensation. Scheduled-dense
calls and prefix-query overwrites use original-dtype PyTorch SDPA.

## 4. Two-step load and memory smoke

Run this before a full generation. A non-50-step run is intentionally restricted
to denoising without decode:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
scripts/run_minimax_h3_demo.sh \
  mpa-sm89-regular2d-mixed \
  /dev/shm/evg-h3-smoke \
  --steps 2 --no-decode
```

The command must exit successfully and write `benchmark.json`,
`run_config.json`, and `denoised_state.pt`. In `benchmark.json`, require:

- `rank_outputs_bit_identical` to be `true`;
- four entries under `per_rank`;
- `status` to be `complete`;
- the reported candidate policy and resource hashes to match this guide.

## 5. Reproduce the three 50-step videos

Use separate output directories and do not change any workload arguments between
candidates:

```bash
mkdir -p results/minimax-h3-sm89-demo

for candidate in dense official-sol mpa-sm89-regular2d-mixed; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  scripts/run_minimax_h3_demo.sh \
    "$candidate" \
    "results/minimax-h3-sm89-demo/$candidate"
done
```

The fixed quality workload is seed 0, 50 scheduler steps, 124 generated frames,
1344x768 output resolution, and the bundled `official-example-1` conditioning.
The exported deliverable is 120 frames at 24 FPS with audio.

Each completed candidate directory contains:

- `out.mp4`: canonical demo video;
- `<candidate>_1344x768_5s_24fps.mp4`: named copy of the same video;
- `denoised_state.pt`: CPU latent artifact for controlled comparisons;
- `benchmark.json`: timing, memory, policy, hashes, and media probe;
- `run_config.json`: compact resolved workload and policy record.

Treat `dense` as the quality baseline. `official-sol` is a reference candidate,
not the baseline. Do not compare videos produced with different checkpoints,
conditioning files, dimensions, frame counts, step counts, seeds, or GPU counts.

## 6. Decode a saved denoised state

If denoising was run with `--no-decode`, decode it later in one process. Do not
use the four-process wrapper for `--decode-only`:

```bash
candidate=mpa-sm89-regular2d-mixed
output_dir=/absolute/path/to/the/candidate/output

CUDA_VISIBLE_DEVICES=0 "$MPA_PYTHON" \
  benchmarks/run_minimax_h3_fp8_ulysses_sm89.py \
  --candidate "$candidate" \
  --output-dir "$output_dir" \
  --diffusers-src "$H3_DIFFUSERS_CHECKOUT" \
  --model-root "$H3_MODEL_ROOT" \
  --checkpoint "$H3_DIT_CHECKPOINT" \
  --decode-only
```

## 7. Optional MPA precision split

The released default is FP8/FP16 = 0.8/0.2. Both overrides must be supplied,
must be positive, and must sum to one:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
scripts/run_minimax_h3_demo.sh \
  mpa-sm89-regular2d-mixed \
  results/minimax-h3-sm89-demo/mpa-custom-split \
  --fp8-ratio 0.7 --fp16-ratio 0.3
```

An overridden split is an ablation and must not be reported as the released
default.

## 8. Common failures

- `set H3_*`: one of the three external resource variables is unset.
- `compressed checkpoint is incomplete` or a SHA256 mismatch: the checkpoint
  is partial or is not the validated compressed DiT.
- `required local sources are missing`: the Diffusers checkout or model root
  does not have the expected layout.
- `world_size=...`: the main runner was not launched with exactly four ranks.
- extension import or architecture errors: rebuild both extensions with the
  active interpreter and `TORCH_CUDA_ARCH_LIST=8.9`.
- out-of-memory during decode: finish denoising with `--no-decode`, release the
  distributed processes, then use the one-GPU `--decode-only` command above.
