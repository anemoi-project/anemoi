# EVG-Project

Efficient Visual Generation (EVG) is an open-source inference framework for visual generation, with a focus on video generation.

This repository combines efficient inference techniques with modern visual generation systems.


## Project Goal

The goal is to build a practical and extensible open-source inference engine
that supports:

- efficient visual (video) generation with high fidelity to dense inference
- long-sequence visual attention acceleration
- mixed-precision attention execution strategies
- inference-serving optimizations


## Draft Attention

Draft Attention is one of the first techniques incorporated into EVG. The broader objective is not limited to one algorithm: EVG is intended to become a general inference engine for visual generation. The implementation is based on the [Draft Attention paper](https://arxiv.org/abs/2505.14708).

## Current Status

EVG is organized as an inference-only runtime. The first implementation layer contains:

- a typed model registry for video generation families
- adapter contracts for external model runtimes
- engine and configuration objects
- placeholders for scheduler, worker, serving, attention, and precision backends
- CLI commands for model discovery and dry-run request planning

Initial model families:

- Wan2.2
- HunyuanVideo-1.5
- LingBot-Video
- LongCat-Video
- Cosmos 3
- SkyReels V3
- Bernini

Install the source package and try the discovery commands:

```bash
python -m pip install -e .
python -m evg.cli.main list-models
python -m evg.cli.main inspect wan2.2 --variant t2v-a14b
python -m evg.cli.main generate \
  --model wan2.2 \
  --variant t2v-a14b \
  --prompt "A cinematic sunrise over a mountain lake" \
  --output outputs/wan22.mp4 \
  --dry-run
```

Run the Draft Attention backend smoke tests:

```bash
scripts/run_wan22_draft_attention_smoke.sh --full-mask-check --compare-dense
scripts/run_hunyuan15_draft_attention_smoke.sh
```

### HunyuanVideo-1.5

Apply the experimental HunyuanVideo-1.5 integration from the root of its
upstream checkout:

```bash
patch -p1 < /path/to/evg/integrations/hunyuanvideo-1.5/patches/evg_draft_attention.patch
```

The integration provides visual-only draft-map guidance, Triton block-sparse
attention, detailed CUDA profiling, and per-step/per-layer sparsity schedules.
Configure the tested project layout and generation settings:

```bash
cd /path/to/evg

export EVG_SERVER_ROOT=/path/to/evg-project
export EVG_MODEL_PATH=/path/to/HunyuanVideo-1.5/snapshots/master
export EVG_VIDEO_LENGTH=121
export EVG_NUM_INFERENCE_STEPS=8
export EVG_SEED=20260731
```

Generate a fully dense baseline:

```bash
EVG_DRAFT_ATTN=false \
EVG_DRAFT_PROFILE=false \
scripts/run_hunyuan15_draft_720p.sh
```

Generate with the standard schedule: the first 25% of diffusion steps are
dense, and the remaining steps use 80% attention sparsity for every layer.

```bash
EVG_DRAFT_ATTN=true \
EVG_DRAFT_SCHEDULE_CONFIG="$PWD/configs/hunyuanvideo-1.5/draft_25dense_80sparse.json" \
scripts/run_hunyuan15_draft_720p.sh
```

Run the per-step/per-layer example:

```bash
EVG_DRAFT_ATTN=true \
EVG_DRAFT_SCHEDULE_CONFIG="$PWD/configs/hunyuanvideo-1.5/draft_layerwise_example.json" \
scripts/run_hunyuan15_draft_720p.sh
```

See [`docs/sparsity_configuration.md`](docs/sparsity_configuration.md) for the
configuration schema, selector syntax, full matrix format, and precedence
rules.

The active acceleration path is sparse BF16 attention. FP8/FP4 mixed precision
is intentionally reserved for future work.

## Todo

- Add mixed-precision policies for long visual sequence attention.
