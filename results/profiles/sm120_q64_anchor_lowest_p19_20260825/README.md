# SM120 Q64 p19 Lowest-Precision Anchor Ablation

Date: 2026-08-25

## Controlled configuration

This run uses the production balanced `24x42 -> 16x63` Q64 partition,
INT8 prefix Q/K/V, an all-INT8 retained phase, the existing 90/85/65
sparsity schedule, and enabled same-frame adjacency anchors. A missing anchor
replaces the weakest ordinary selection in the lowest active precision budget;
for this configuration that budget is INT8. The retained-edge count is not
expanded.

The workload is p19, seed 0, 243 generated / 239 delivered frames,
1344x768, 50 steps / 49 DiT evaluations. The matching p19 Dense output is the
only quality reference. Sol-Attention and SpargeAttention are comparators, not
accuracy references.

## Dense-relative quality

`Global PSNR` is the repository `benchmarks/minimax_h3_50case/metrics.py`
metric: one RGB MSE accumulated over all aligned frames. `Frame-mean PSNR` and
`worst frame` reproduce the historical `video_psnr()` rule, which computes
PSNR per frame before averaging. `Latent cosine` compares the flattened final
video latent with the matching Dense latent.

| p19 candidate | Global RGB MSE | Global PSNR (dB) | Frame-mean PSNR (dB) | Worst frame (dB) | Latent cosine |
|---|---:|---:|---:|---:|---:|
| Sol-Attention | 1142.749586 | 17.551293 | 18.025 | 15.730 | 0.889747 |
| SpargeAttention | 828.378737 | 18.948514 | 19.010 | 16.605 | 0.926463 |
| Q128 INT8 | 650.311732 | 19.999588 | 20.059 | 16.860 | 0.931352 |
| Q128 NV60/INT40 | 1344.007467 | 16.846787 | 17.211 | 15.264 | 0.878497 |
| Q128 NV75/INT25 | 1484.791396 | 16.414149 | 16.752 | 14.747 | 0.868864 |
| Q64 old partition, no anchors | 1929.715933 | 15.275870 | 15.887 | 11.046 | 0.834760 |
| Q64 balanced `16x63`, no anchors | 1203.754159 | 17.325426 | 18.007 | 15.252 | 0.886795 |
| **Q64 balanced `16x63`, anchors in INT8 budget** | **644.365297** | **20.039482** | **20.106** | **17.411** | **0.934257** |

Relative to balanced Q64 without anchors, the new anchor rule improves global
PSNR by `+2.714057 dB`, raises latent cosine by `+0.047462`, and reduces RGB
MSE by `46.47%`. Relative to the old unbalanced/no-anchor Q64 control, the
improvements are `+4.763612 dB`, `+0.099497` cosine, and `66.61%` lower MSE.

The current result is also `+0.039894 dB` / `+0.002905` cosine above the Q128
all-INT8 control, `+1.090968 dB` / `+0.007794` above SpargeAttention, and
`+2.488189 dB` / `+0.044510` above Sol-Attention.

## Earlier Q64 cosine-only diagnostics

These runs changed additional topology or precision variables and therefore
are context rather than controlled rows in the table above:

| Earlier p19 diagnostic | Latent cosine vs Dense |
|---|---:|
| Old Q64 anchor-only, FP16 prefix | 0.879460 |
| Old Q64 public-like anchors + FP16 prefix + 80/20 INT8/FP16 | 0.936483 |
| Released regular2D topology and policy | 0.943865 |
| Regular2D topology with current precision policy | 0.927379 |
| Regular2D native-prefix repair candidate | 0.942967 |

## Artifacts

- Output: `outputs/sm120_q64_anchor_lowest_p19_20260825/p19_q64_int8_anchor/`
- `benchmark.json`: SHA256 `49c1847cb0e16028ea9063216a5fac60f4c40aa34ab8c741ab4596461f8bd61d`
- `run_config.json`: SHA256 `de9ebab4b8fe6e2748fc7278e3018312fbf280789498a07f1202ce2e83462252`
- `out.mp4`: SHA256 `ea852f1ecb83289aa18865fd83fd82fc9a5a22552eaa0041bbf93b57d8fb5309`
- `denoised_state.pt`: SHA256 `c715e2876bd2565d547975cd7940a518c6acb042c649f66ca4187f790828fda4`
- `quality_vs_dense.json`: SHA256 `b428507de6e482f2547b8167d3fcbabc862f9ce43c99b13b09404da2886d570c`

The benchmark reports `status=complete`, 49/49 DiT evaluations, 1872 MPA
calls, bit-identical rank output, and `enable_anchors=true`.

## Compact native Anchor component result

The later performance correction was evaluated only at the route/materialize
component boundary, as requested; the interrupted replacement E2E run stopped
at 36/49 and produced no result artifact. The input is real Q/K/V from the
matching p19 Dense step10/layer2 capture (SHA256 `42617124...`). Dense is the
numerical/input baseline but has no DraftMap routing component, so native
no-anchor is the isolated Anchor-overhead control.

The request-static layout now caches 5,402 compact anchor IDs. The existing CUB
global sort and precision scatter are unchanged; the new CUDA projection scans
only those IDs and then walks the existing lowest-INT8 tail for victims. It
does not run a second sort, runtime `nonzero`, eager/PyTorch correction, or an
alternate materializer.

Five warmups plus 31 forward/reverse interleaved CUDA-event samples measured:

| p19 Q64 route implementation | Route only (median ms) | Materialize only (median ms) | Combined (median ms) |
|---|---:|---:|---:|
| Native, no anchors | 5.0285 | 0.2422 | 5.3048 |
| Previous native Anchor, per-head `R^2` scan | 7.1147 | 0.2397 | 7.3689 |
| **Compact native Anchor** | **5.1080** | **0.2397** | **5.3275** |
| Historical eager Anchor reference | 12.4888 | 1.3520 | 14.0923 |

Compact Anchor adds `0.0228 ms/call` (`+0.429%`) to combined native
route/materialization in this run, versus `2.0782 ms/call` for the previous
`R^2` scan. That removes `98.91%` of the measured Anchor increment and is
`2.645x` faster than the eager Anchor boundary. The purely mechanical
1,872-call projection is `+0.0426 s`; no E2E timing claim is made from it.

At requested sparsity `90%`, `85%`, and `65%`, measured effective sparsity is
respectively `0.900000293`, `0.849999707`, and `0.650000293`. Compact native and
Python reference have exact active IDs, per-phase counts, retained budgets, and
physical materialized active IDs at all three points. Full machine-readable
evidence is in `component_route_performance.json`.
