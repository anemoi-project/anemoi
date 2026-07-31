# EVG Architecture

EVG is an inference-only engine for visual generation. The first target is
video generation, especially long-sequence video generation. Training,
fine-tuning, and dataset pipelines are intentionally outside the core project.

## System Shape

```text
client or CLI
  -> serve/API layer
  -> EVGEngine
  -> scheduler
  -> worker
  -> model runner
  -> visual pipeline components
  -> attention and precision backends
```

## Package Layout

```text
evg/
  cli/             command line tools
  config/          engine, precision, and attention configuration
  engine/          public engine entrypoint
  layers/          attention and mixed precision backends
  model_runner/    denoising loops and model forward dispatch
  models/          model registry and family adapters
  pipeline/        visual generation pipeline graph
  scheduler/       request batching and cancellation
  serve/           HTTP/OpenAI-compatible serving surfaces
  worker/          GPU worker ownership and execution loops
```

Top-level directories:

```text
benchmarks/       reproducible performance records
csrc/             future C++/CUDA/Triton extension sources
docs/             architecture, model support, and roadmap docs
examples/         runnable generation examples
tests/            unit and smoke tests
```

## First Design Rule

Model support enters through adapters or small integration patches. Draft
Attention, sparsity scheduling, future mixed precision, cache, and request
scheduling remain EVG-owned features rather than model-local implementations.

## Draft Attention Backend

The initial Draft Attention implementation is inference-only and has two
separable pieces:

- draft-map construction from pooled Q/K, using blockwise online softmax
- block-sparse attention over Q/K/V from the generated block mask

The Torch path is the correctness reference. The Triton path includes QK-only
online-softmax draft-map kernels and indexed block-sparse attention. A resolved
step-by-layer matrix separates optimization policy from model and kernel code.

The current accelerated production path is sparse BF16 attention. Precision
policy types reserve the future FP8/FP4 boundary but are not active yet.

## Support Levels

- `metadata-only`: model is listed, but no adapter exists.
- `adapter-scaffolded`: EVG knows the model family, variants, tasks, and runtime.
- `experimental`: an adapter can run a smoke generation path.
- `supported`: tested path with documented commands, benchmarks, and CI coverage.
