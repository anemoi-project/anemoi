# Draft Attention

EVG's current Draft Attention backend has three pieces:

1. Reshape visual Q/K into the latent video grid and apply 2D average pooling.
2. Build a draft map with blockwise QK, online softmax statistics, and global top-k.
3. Reorder Q/K/V into spatial blocks and run Triton block-sparse attention.

The draft-map builder does not compute `Q @ K.T` as one giant matrix. It uses
blockwise QK tiles and online softmax row statistics, then makes a headwise
top-k block mask from the resulting softmax probabilities.

Only visual tokens enter draft-map guidance. Visual/text and text/text
interactions remain dense. The output is restored to the model's original token
order after sparse attention.

The sparse attention implementation has:

- a Torch reference path for correctness
- Triton draft-map and indexed block-sparse attention kernels
- CUDA-event profiling for draft map, reorder, sparse attention, and restore
- per-diffusion-step and per-model-layer sparsity scheduling

See [sparsity_configuration.md](sparsity_configuration.md) for the schedule
schema and precedence rules.

## Smoke Tests

Tiny CPU-safe checks:

```bash
scripts/run_wan22_draft_attention_smoke.sh --full-mask-check --compare-dense
scripts/run_hunyuan15_draft_attention_smoke.sh
```

Full-shape CUDA checks:

```bash
EVG_FULL_SHAPE=1 EVG_DEVICE=cuda EVG_DTYPE=bfloat16 EVG_DRAFT_BACKEND=auto scripts/run_wan22_draft_attention_smoke.sh
EVG_FULL_SHAPE=1 EVG_DEVICE=cuda EVG_DTYPE=bfloat16 EVG_DRAFT_BACKEND=auto scripts/run_hunyuan15_draft_attention_smoke.sh
```

The HunyuanVideo-1.5 integration patch and 720p launcher are available under
`integrations/hunyuanvideo-1.5` and `scripts/run_hunyuan15_draft_720p.sh`.
Wan2.2 remains at backend smoke-test support.

## Presets

| Preset | Visual tokens | Text tokens | Heads | Head dim | Default sparsity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wan2.2-480p` | 32,256 | 0 | 40 | 128 | 0.75 |
| `wan2.2-720p` | 80,640 | 0 | 40 | 128 | 0.75 |
| `hunyuanvideo-1.5-480p` | 50,688 | 256 | 24 | 128 | 0.90 |
| `hunyuanvideo-1.5-720p` | 126,720 | 256 | 24 | 128 | 0.90 |
