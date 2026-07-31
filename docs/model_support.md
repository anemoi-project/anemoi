# Model Support Plan

This file records the initial video model families EVG will target.

| Family | Default variant | Runtime | Current EVG status | Main reason to support |
| --- | --- | --- | --- | --- |
| Wan2.2 | `t2v-a14b` | Diffusers/native | `adapter-scaffolded` | Major open video family and a natural first Draft Attention target. |
| HunyuanVideo-1.5 | `480p-t2v` | Diffusers/native | `adapter-scaffolded` | Lightweight 8.3B baseline with sparse attention and cache references. |
| LingBot-Video | `dense-1.3b` | Diffusers/FastVideo | `adapter-scaffolded` | Dense and MoE embodied-video family for scaling tests. |
| LongCat-Video | `13.6b` | Diffusers | `adapter-scaffolded` | Long-video native tasks and block sparse attention upstream. |
| Cosmos 3 | `nano` | Omni/Diffusers | `adapter-scaffolded` | World-model generator with video, audio, and action conditioning. |
| SkyReels V3 | `r2v-14b` | Diffusers/upstream script | `adapter-scaffolded` | Reference-to-video, V2V, and talking-avatar workloads. |
| Bernini | `renderer-wan22` | Hybrid upstream script | `adapter-scaffolded` | Planner-renderer architecture built around Wan2.2 components. |

## Near-Term Milestones

1. Make every family visible through `evg list-models` and `evg inspect`.
2. Add one runnable single-GPU smoke path for Wan2.2 T2V.
3. Add the same smoke path for HunyuanVideo-1.5 480P T2V.
4. Introduce EVG attention backend selection with `dense` as baseline.
5. Add Draft Attention as a selectable backend for the first compatible DiT block.
6. Add benchmark records for latency, memory, frame count, resolution, and seed.
