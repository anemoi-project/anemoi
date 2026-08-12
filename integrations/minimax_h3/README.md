# MiniMax-H3 adapter

This directory contains the adapter used by the EVG demo runner.  The released
profile is intentionally fixed to the SM89 regular-2D path:

- request-level selection between the validated 8x7 and 8x8 logical tiles;
- physical Q64xK64 attention on RTX 4090;
- DraftMap row-softmax probability and exact per-head global top-k;
- legal same-frame 2-D cross anchors within the unchanged route budget;
- retained FP8/FP16 work split 80/20;
- original-dtype SDPA for scheduled-dense calls and prefix-query rows;
- no skip compensation.

Use `scripts/run_minimax_h3_demo.sh`; the runner exposes only `dense`,
`official-sol`, and `mpa-sm89-regular2d-mixed`.

See [REPRODUCTION.md](REPRODUCTION.md) for the pinned resources, full build
procedure, controlled three-candidate run, output contract, and troubleshooting.

The checkpoint, MiniMax-H3 Diffusers model, and Diffusers source checkout are
external resources. Build and run with:

```bash
python -m pip install -r requirements-demo.txt
scripts/build_router_cuda.sh
scripts/build_attention_cuda.sh

export H3_DIFFUSERS_CHECKOUT=/path/to/diffusers
export H3_MODEL_ROOT=/path/to/MiniMax-H3-diffusers
export H3_DIT_CHECKPOINT=/path/to/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
scripts/run_minimax_h3_demo.sh mpa-sm89-regular2d-mixed /dev/shm/h3-mpa-demo
```

Set `MPA_PYTHON` or `MPA_CUDA_HOME` when the desired interpreter or CUDA
toolkit is not the shell default. All compile/runtime caches default to
`/dev/shm`.
