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

Tiny CPU-safe synthetic checks:

```bash
scripts/run_wan22_draft_attention_smoke.sh --full-mask-check --compare-dense
scripts/run_hunyuan15_draft_attention_smoke.sh
```

CUDA full-shape checks:

```bash
EVG_FULL_SHAPE=1 EVG_DEVICE=cuda EVG_DTYPE=bfloat16 EVG_DRAFT_BACKEND=auto scripts/run_wan22_draft_attention_smoke.sh
EVG_FULL_SHAPE=1 EVG_DEVICE=cuda EVG_DTYPE=bfloat16 EVG_DRAFT_BACKEND=auto scripts/run_hunyuan15_draft_attention_smoke.sh
```
