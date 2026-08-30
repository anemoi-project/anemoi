# Benchmarks

Benchmark records should make optimization changes comparable across models.

Each benchmark should report:

- model family and variant
- task
- resolution, frame count, FPS, and seed
- attention backend
- dense-step fraction and resolved per-step/per-layer sparsity policy
- precision policy
- diffusion-only dense and sparse latency
- draft-map, reorder, sparse-kernel, and restore latency
- peak allocated and reserved GPU memory
- prompt, seed, and initial-latent fingerprint
- output artifact path

## MiniMax-H3 benchmark entry point

The current end-to-end benchmark entry point is `scripts/run_minimax_h3.sh`.
See the stable [attention API](../docs/attention_api.md) for the supported
native MPA contract and the
[MiniMax-H3 reproduction guide](../anemoi/models/minimax_h3/REPRODUCTION.md) for
the controlled benchmark procedure and output contract.
