# HunyuanVideo-1.5 Integration

This integration keeps EVG-owned kernels and scheduling in the `evg` package.
The upstream model receives only thin hooks for:

- Draft Attention CLI and inference state
- diffusion-step schedule resolution
- stable global attention-layer indices
- routing dense or sparse attention for each step/layer cell
- diffusion-only latency, memory, and Draft Attention stage profiling
- initial-latent fingerprint logging for reproducible comparisons

Apply the patch from the root of a compatible HunyuanVideo-1.5 checkout:

```bash
patch -p1 < /path/to/evg/integrations/hunyuanvideo-1.5/patches/evg_draft_attention.patch
```

The patch requires EVG to be installed or importable in the Hunyuan Python
environment. Quantized attention is intentionally outside the current
integration scope.
