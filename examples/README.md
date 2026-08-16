# Examples

The current executable model path is MiniMax-H3:

```bash
scripts/run_minimax_h3.sh
```

The annotated released MPA configuration is
[`minimax-h3/mpa-ragged2d-mixed.yaml`](minimax-h3/mpa-ragged2d-mixed.yaml).
Reference it explicitly from the repository root with:

```bash
scripts/run_minimax_h3.sh \
  mpa-ragged2d-mixed \
  outputs/minimax-h3/mpa \
  --mpa-config examples/minimax-h3/mpa-ragged2d-mixed.yaml
```

Copy the YAML before editing it when creating a custom experiment policy.
