# MiniMax-H3 model path

This package contains Anemoi's current end-to-end model mainline. Both
architecture defaults use Q64 stripe-compact routing, global DraftMap ranking,
and a fixed 80% Mean-only sparse-layer budget. SM89 and SM120 use the same
pure-INT8 policy for prefix queries, prefix K/V, and routed video attention;
runtime device capability selects the native backend.

The shared policy also uses:

- exact-cover stripe-compact ragged 2-D blocks at logical capacity 64;
- physical Q64xK64 native attention on SM89;
- pooled-QK row-softmax probability with exact per-head global top-k;
- uniform 80% sparsity for every sparse layer;
- original-dtype SDPA for scheduled-dense calls;
- direct dropping of unselected block pairs.

SM89 and SM120 Q64/Q128 can opt into Mean/MaxPool probability fusion with
`maxpool_weight` in `[0, 1]`; the default `0` keeps Mean-only routing.

The same model runner exposes `dense`, `official-sol`, and
`mpa-ragged2d-mixed` so the attention implementation is the controlled
variable.

From the repository root, run:

```bash
scripts/run_minimax_h3.sh
```

The launcher applies the canonical Q64 pure-INT8 policy automatically. To
reference an architecture-named compatibility alias explicitly:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa-sm120-q64 \
  --mpa-config examples/minimax-h3/mpa-sm120-q64-int8.yaml
```

Selected GPUs must all be SM89 or all be SM120. The runtime dispatches the
matching native backend. SM89 uses the public INT8/FP16 configurations, while
NVFP4 configurations require SM120 GPUs.

See [REPRODUCTION.md](REPRODUCTION.md) for the pinned resources, manual build,
multi-GPU selection, smoke test, and output contract.
