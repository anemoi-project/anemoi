# Sparse Attention Configuration

> **Scope:** This document describes the legacy/generic BF16 Draft Attention
> configuration. For the stable native mixed-precision attention API, see
> [Attention API](attention_api.md).

Anemoi resolves sparse-attention configuration into an immutable matrix with one
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

## Precision Scope

This configuration object belongs to the generic BF16 Draft Attention backend.
MiniMax-H3 MPA uses its fixed 80% default, optional layer-band overrides, and
architecture-neutral pure-INT8 policy directly in
`anemoi.models.minimax_h3.runner`; it does not consume this legacy JSON schema.
