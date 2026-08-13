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

## Draft Attention Smoke Commands

The current end-to-end benchmark entry point is `scripts/run_minimax_h3.sh`.
