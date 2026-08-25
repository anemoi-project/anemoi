# Examples

The current executable model path is MiniMax-H3:

```bash
scripts/run_minimax_h3.sh
```

The launcher selects Q64 FP8/FP16 on SM89 and Q64 pure INT8 on SM120. The
corresponding annotated configurations are
[`minimax-h3/mpa-ragged2d-mixed.yaml`](minimax-h3/mpa-ragged2d-mixed.yaml) and
[`minimax-h3/mpa-sm120-q64-int8.yaml`](minimax-h3/mpa-sm120-q64-int8.yaml).
Reference the SM89 policy explicitly from the repository root with:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa \
  --mpa-config examples/minimax-h3/mpa-ragged2d-mixed.yaml
```

Copy the YAML before editing it when creating a custom experiment policy.

The Anchor-enabled SM120 Q64 pure-INT8 default can be stated explicitly with:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa-sm120-q64 \
  --mpa-config examples/minimax-h3/mpa-sm120-q64-int8.yaml
```
