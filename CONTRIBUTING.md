# Contributing

EVG is an inference-only visual generation engine. Contributions should keep
training, fine-tuning, and dataset pipelines outside the core runtime.

## Development Setup

```bash
pip install -e .[dev]
python -m unittest discover -s tests
```

Install runtime dependencies only when working on executable model paths:

```bash
pip install -e .[runtime]
```

## Code Organization

- Model metadata and variants belong in `evg/models/catalog.py`.
- Family-specific runtime glue belongs in `evg/models/adapters/`.
- Shared denoising execution belongs in `evg/model_runner/`.
- Attention implementations belong in `evg/layers/attention/`.
- Precision and quantization policies belong in `evg/layers/precision/`.
- Request batching belongs in `evg/scheduler/`.
- GPU process ownership belongs in `evg/worker/`.
- HTTP APIs belong in `evg/serve/`.

## Support Levels

Do not mark a model as `supported` until it has:

- a documented command
- a reproducible smoke output
- latency and memory measurements
- tests for request validation and adapter selection
