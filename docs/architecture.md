# Anemoi Architecture

Anemoi is an inference-only engine for visual generation. The first target is
video generation, especially long-sequence video generation. Training,
fine-tuning, and dataset pipelines are intentionally outside the core project.

## System shape

```text
client or CLI
  -> serve/API layer
  -> AnemoiEngine
  -> scheduler
  -> worker
  -> model runner
  -> model adapter
  -> anemoi_attention
  -> generic MPA executor and ragged 2-D routing
  -> SM89 or SM120 Q64/Q128 native backend
```

The model adapter owns model-specific scheduling and calibration. The stable
`anemoi_attention` interface accepts BSHD Q/K/V tensors plus a packed
prefix/video layout, and the generic MPA executor builds the request-static
layout, routes retained block pairs, and dispatches by CUDA capability and
query-block size. The selected native backend returns output in the caller's
original token order.

## Package layout

```text
anemoi/
  cli/             command line tools
  config/          engine, precision, and attention configuration
  engine/          public engine entrypoint
  layers/          stable attention API, MPA executor, routing, and backends
  model_runner/    denoising loops and model forward dispatch
  models/          model registry and family adapters
  pipeline/        visual generation pipeline graph
  scheduler/       request batching and cancellation
  serve/           HTTP/OpenAI-compatible serving surfaces
  worker/          GPU worker ownership and execution loops
```

Top-level directories:

```text
csrc/             current native CUDA implementations for SM89 and SM120
docs/             architecture, model support, and roadmap docs
examples/         runnable generation examples
tests/            unit and smoke tests
```

## Native mixed-precision attention

SM89 Q64/Q128 and SM120 Q64/Q128 are the current production native paths. They
support structured visual-only and packed prefix-plus-video attention. The
stable precision cells cover INT8 and FP16 on SM89, and NVFP4, INT8, and FP16
combinations on SM120. Runtime validation rejects unsupported architecture,
query geometry, prefix precision, and retained-block precision combinations
before kernel launch.

The ragged 2-D partitioner adapts to the runtime spatial grid and physical block
capacity. It creates the minimum number of connected blocks, balances real-token
counts globally within one token, evaluates both spatial orientations, and
selects the best compact or legacy candidate by its deterministic shape cost.
Partition metadata is cached per geometry; it is not tied to a model or a
hard-coded resolution table.

## Model integration

Model support enters through adapters or small integration patches. A compatible
visual model can call the generic attention API when it can provide BSHD Q/K/V,
the prefix/video layout, and the transformer-layer index. Dense-first scheduling
and model-owned calibration remain adapter responsibilities.

MiniMax-H3 is the currently validated end-to-end model family on RTX 4090 and
RTX 5090. Other compatible visual models can integrate the generic API without
depending on the MiniMax-H3 runtime.

## Legacy Draft Attention backend

The generic BF16 Draft Attention implementation remains as framework
infrastructure and a correctness/reference path. It has two separable pieces:

- draft-map construction from pooled Q/K, using blockwise online softmax
- block-sparse attention over Q/K/V from the generated block mask

Its historical step-by-layer sparse schema is separate from the stable native
MPA configuration. See [Draft Attention](draft_attention.md) for the legacy
design and [Attention API](attention_api.md) for the current integration
contract.

## Support levels

- `metadata-only`: model is listed, but no adapter exists.
- `adapter-scaffolded`: Anemoi knows the model family, variants, tasks, and runtime.
- `experimental`: an adapter can run a smoke generation path.
- `supported`: tested path with documented commands, benchmarks, and CI coverage.
