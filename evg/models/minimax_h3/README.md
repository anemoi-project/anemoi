# MiniMax-H3 model path

This package contains EVG's current end-to-end model mainline. The released MPA
profile uses:

- exact-cover stripe-compact ragged 2-D blocks at logical capacity 64;
- physical Q64xK64 native attention on SM89;
- pooled-QK row-softmax probability with exact per-head global top-k;
- same-frame ragged adjacency anchors, with a minimum-feasible budget fallback
  for very small grids;
- a retained-block FP8/FP16 split of 80/20;
- original-dtype SDPA for scheduled-dense calls and prefix-query rows;
- no skip compensation.

The same model runner exposes `dense`, `official-sol`, and
`mpa-ragged2d-mixed` so the attention implementation is the controlled
variable.

From the repository root, run:

```bash
scripts/run_minimax_h3.sh
```

The launcher loads the annotated policy at
[`examples/minimax-h3/mpa-ragged2d-mixed.yaml`](../../../examples/minimax-h3/mpa-ragged2d-mixed.yaml)
by default. To reference a configuration explicitly:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa \
  --mpa-config examples/minimax-h3/mpa-ragged2d-mixed.yaml
```

See [REPRODUCTION.md](REPRODUCTION.md) for the pinned resources, manual build,
multi-GPU selection, smoke test, and output contract.
