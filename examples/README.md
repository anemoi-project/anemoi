# Examples

The current executable model path is MiniMax-H3:

```bash
scripts/run_minimax_h3.sh
```

The launcher applies one Q64 pure-INT8 policy on SM89 and SM120, including
prefix Q/K/V and routed video attention. The canonical configuration is
[`minimax-h3/mpa-ragged2d-mixed.yaml`](minimax-h3/mpa-ragged2d-mixed.yaml).
[`minimax-h3/mpa-sm120-q64-int8.yaml`](minimax-h3/mpa-sm120-q64-int8.yaml)
is a compatibility alias with the same parsed policy. Reference it explicitly
from the repository root with:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa \
  --mpa-config examples/minimax-h3/mpa-ragged2d-mixed.yaml
```

Copy the YAML before editing it when creating a custom experiment policy.

The SM120-named alias is equivalent:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa-sm120-q64 \
  --mpa-config examples/minimax-h3/mpa-sm120-q64-int8.yaml
```

The canonical and architecture-alias profiles use fixed 80% sparsity in every
sparse layer. The Q128 files are minimal overrides that inherit the same empty
layer-band schedule and global `sparsity_ratio=0.80` from `SparseConfig`.
