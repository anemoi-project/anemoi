# Model Support Plan

This file records the initial video model families EVG will target.

| Family | Default variant | Runtime | Current EVG status | Main reason to support |
| --- | --- | --- | --- | --- |
| MiniMax-H3 | `fl2va-pruned-fp8` | Diffusers/Sol/native MPA | `supported` | Current end-to-end mainline with reproducible Dense, official Sol, and mixed-precision attention paths. |
| Wan2.2 | `t2v-a14b` | Diffusers/native | `adapter-scaffolded` | Major open video family and a natural first Draft Attention target. |
| LingBot-Video | `dense-1.3b` | Diffusers/FastVideo | `adapter-scaffolded` | Dense and MoE embodied-video family for scaling tests. |
| LongCat-Video | `13.6b` | Diffusers | `adapter-scaffolded` | Long-video native tasks and block sparse attention upstream. |
| Cosmos 3 | `nano` | Omni/Diffusers | `adapter-scaffolded` | World-model generator with video, audio, and action conditioning. |
| SkyReels V3 | `r2v-14b` | Diffusers/upstream script | `adapter-scaffolded` | Reference-to-video, V2V, and talking-avatar workloads. |
| Bernini | `renderer-wan22` | Hybrid upstream script | `adapter-scaffolded` | Planner-renderer architecture built around Wan2.2 components. |

## Near-Term Milestones

1. Make every family visible through `evg list-models` and `evg inspect`.
2. Keep MiniMax-H3 Dense as the quality baseline for every attention change.
3. Optimize the native MiniMax-H3 MPA path without changing its routing contract.
4. Extend the same controlled runner to additional validated GPU architectures.
