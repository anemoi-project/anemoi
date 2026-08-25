# SM120 Q64 p01 Lowest-Precision Anchor Transfer Check

Date: 2026-08-25

## Configuration

This run reuses the validated p19 configuration without policy changes:
production balanced `24x42 -> 16x63` Q64 partition, INT8 prefix Q/K/V,
all-INT8 retained phase, 90/85/65 sparsity schedule, and same-frame adjacency
anchors charged to the fixed INT8 budget. The workload is p01, seed 0,
243 generated / 239 delivered frames, 1344x768, and 50 steps / 49 DiT
evaluations.

The matching p01 Dense output is the only quality reference. Sol-Attention and
SpargeAttention are comparators, not accuracy references.

## Dense-relative quality

| p01 candidate | Global RGB MSE | Global PSNR (dB) | Historical frame-mean PSNR (dB) | Worst frame (dB) | Latent cosine |
|---|---:|---:|---:|---:|---:|
| Sol-Attention | 370.473507 | 22.443232 | 24.002 | 16.983 | 0.972538 |
| SpargeAttention | 608.942162 | 20.285043 | 21.514 | 14.715 | 0.959871 |
| Q128 INT8 | 885.719294 | 18.657843 | 21.274 | 10.902 | 0.953809 |
| Q64 old partition, no anchors | 930.302762 | 18.444561 | 21.692 | 10.654 | 0.951619 |
| **Q64 balanced `16x63`, anchors in INT8 budget** | **914.933219** | **18.516910** | **20.880** | **13.103** | **0.950451** |

Relative to the existing Q64 p01 control, the new combined production
configuration improves global PSNR by `+0.072349 dB`, reduces global RGB MSE
by `1.65%`, and improves the worst frame by `+2.449191 dB`. Historical
frame-mean PSNR decreases by `0.811669 dB`, and latent cosine decreases by
`0.001167`.

The current result is `0.140933 dB` and `0.003358` cosine below the Q128 INT8
control. It does not reproduce the severe old-Q64 p19 degradation, but the
large p19 gain does not transfer as a uniform p01 gain.

There is no balanced/no-anchor p01 run, so this p01 comparison validates the
combined balanced-partition plus anchor policy; it does not isolate the anchor
change by itself.

## Artifacts

- Output: `outputs/sm120_q64_anchor_lowest_p01_20260825/p01_q64_int8_anchor/`
- `benchmark.json`: SHA256 `d8f4e358b7702f44e0c475a08d386809ecac2acfd1cfdfc3d28f7c8f2fe1b32c`
- `run_config.json`: SHA256 `df5bd60eeed79f4471437ae9b86f3d9779731bddb65a1265d1c403c32a9bf027`
- `out.mp4`: SHA256 `9dba6d2af17a8b85f78434e2d4a65aa2bc978d72330f8d0f2d3846c7f328d92e`
- `denoised_state.pt`: SHA256 `132abe12c258e210063d035fd9cb67105988f49514de143e156e677c7c9c898e`
- `quality_vs_dense.json`: SHA256 `3e199640e6272e05278410b68eecd80d5157cf5892f5a33b2e525477e899076c`

The benchmark records `status=complete`, 49/49 DiT evaluations, 1872 MPA
calls, bit-identical rank output, and `enable_anchors=true`.
