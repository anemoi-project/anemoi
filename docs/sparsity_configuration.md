# Sparse Attention Configuration

EVG resolves sparse-attention configuration into an immutable matrix with one
row per diffusion step and one column per model attention layer. A value of
`0.0` selects dense attention for that cell. Values greater than zero select
Draft Attention with that sparsity ratio.

## Compact Configuration

```json
{
  "enabled": true,
  "dense_step_fraction": 0.25,
  "default_sparsity": 0.8,
  "rules": [
    {
      "steps": "2-3",
      "layers": "0-8",
      "sparsity": 0.7
    },
    {
      "steps": "6-7",
      "layers": "45-53",
      "sparsity": 0.85
    }
  ]
}
```

`steps` and `layers` accept a zero-based integer, a sequence of integers, `*`,
or a compact selector such as `0,2-5,9`. Rules are applied in order, so later
rules override earlier rules for overlapping cells.

Resolution precedence is:

1. `default_sparsity` initializes every cell.
2. `sparsity_matrix`, when present, replaces those defaults.
3. Ordered `rules` override selected cells.
4. `dense_step_fraction` forces the initial diffusion steps to `0.0`.

The dense prefix always wins. For eight steps, a value of `0.25` makes steps 0
and 1 fully dense, regardless of matrix values or rules.

## Full Matrix

For complete control, provide a matrix directly. Its shape must be exactly
`total_steps x num_attention_layers` at runtime.

```json
{
  "enabled": true,
  "dense_step_fraction": 0.0,
  "default_sparsity": 0.0,
  "sparsity_matrix": [
    [0.0, 0.0, 0.0],
    [0.5, 0.6, 0.7],
    [0.8, 0.8, 0.9]
  ]
}
```

All sparsity values must be in `[0, 1)`. Step and layer selectors are validated
against the loaded model before diffusion starts.

## HunyuanVideo-1.5

The integration assigns global layer IDs to double-stream blocks first and
single-stream blocks second. The checked 720p T2V checkpoint has 54 attention
layers, indexed `0-53`.

The standard schedule is:

```text
configs/hunyuanvideo-1.5/draft_25dense_80sparse.json
```

Run a custom schedule with:

```bash
EVG_DRAFT_SCHEDULE_CONFIG=/path/to/schedule.json \
  scripts/run_hunyuan15_draft_720p.sh
```

The legacy `--draft_dense_fraction` and `--draft_sparsity_ratio` options remain
as a uniform fallback when no JSON schedule is supplied.

## Precision Scope

The current acceleration pipeline is BF16 Draft Attention with Triton sparse
attention. FP8/FP4 and mixed-precision policies remain future work and are not
part of this schedule.
