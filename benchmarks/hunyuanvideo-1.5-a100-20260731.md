# HunyuanVideo-1.5 A100 Benchmark

## Environment

- GPU: one NVIDIA A100 PCIe 40 GB
- PyTorch: 2.7.1+cu126
- Triton: 3.3.1
- FlashAttention: 2.8.3.post1
- dtype: BF16
- resolution: 1280 x 720
- frames: 121 at 24 FPS
- diffusion steps: 8
- seed: 20260731
- initial latent SHA-256:
  `a9e3e54b4cc42c04402f5e07e50750db66459b992b93336452a01f48cdfaf828`

The transformer remained resident on GPU during timing. Model loading,
offloading, text encoding, VAE decoding, and video writing were excluded.

## Schedule

- steps 0-1: fully dense
- steps 2-7: 80% visual sparsity for all 54 attention layers
- text-related attention: dense
- quantization, cache, SR, and torch compilation: disabled

Configuration:
`configs/hunyuanvideo-1.5/draft_25dense_80sparse.json`

## Results

| Mode | Diffusion latency | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: |
| Dense FlashAttention | 656.677 s | 32.007 GiB | 36.342 GiB |
| EVG Draft Attention | 422.680 s | 32.009 GiB | 38.375 GiB |

- speedup: 1.554x
- latency reduction: 35.63%
- time saved: 233.997 s

## Draft Attention Profile

The six sparse steps produced 324 calls: 54 layers times 6 steps.

| Stage | Total | Per call |
| --- | ---: | ---: |
| Draft map | 2.208 s | 6.816 ms |
| Reorder | 6.989 s | 21.572 ms |
| Sparse attention | 144.868 s | 447.122 ms |
| Restore | 2.260 s | 6.975 ms |
| Framework overhead | 0.009 s | 0.028 ms |
| Total | 156.334 s | 482.514 ms |

Additional overhead excluding the sparse-attention kernel was 11.467 seconds,
or 2.71% of sparse diffusion latency.

## Artifacts

```text
hunyuan15_dense_20260731_083513_same_seed_seed20260731_720p_dunhuang_121f_8steps.mp4
hunyuan15_draft_25dense_80sparse_triton_20260731_084842_same_seed_seed20260731_720p_dunhuang_121f_8steps.mp4
```

Both outputs were verified as 121-frame, 1280 x 720, 24 FPS H.264 videos.
