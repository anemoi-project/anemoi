# MiniMax-H3 model path

This package contains Anemoi's current end-to-end model mainline. Both
architecture defaults use Q64 stripe-compact routing, global DraftMap ranking,
and same-frame anchors charged to the fixed lowest-precision budget. SM89 uses
FP8/FP16 80/20; SM120 uses pure INT8 for prefix queries, prefix K/V, and routed
video attention.

The shared policy also uses:

- exact-cover stripe-compact ragged 2-D blocks at logical capacity 64;
- physical Q64xK64 native attention on SM89;
- pooled-QK row-softmax probability with exact per-head global top-k;
- same-frame ragged adjacency anchors charged to the lowest configured
  precision budget without increasing the retained edge count;
- original-dtype SDPA for scheduled-dense calls;
- no skip compensation.

The same model runner exposes `dense`, `official-sol`, and
`mpa-ragged2d-mixed` so the attention implementation is the controlled
variable.

From the repository root, run:

```bash
scripts/run_minimax_h3.sh
```

The launcher selects the annotated SM89 or SM120 Q64 policy automatically. To
reference the SM120 pure-INT8 configuration explicitly:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa-sm120-q64 \
  --mpa-config examples/minimax-h3/mpa-sm120-q64-int8.yaml
```

Selected GPUs must all be SM89 or all be SM120. The runtime dispatches the
matching native backend and rejects an SM120-only policy on SM89.

See [REPRODUCTION.md](REPRODUCTION.md) for the pinned resources, manual build,
multi-GPU selection, smoke test, and output contract.
