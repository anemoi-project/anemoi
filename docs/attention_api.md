# Attention API

_Stable inference contract for structured visual self-attention on SM89 and SM120_

---

Anemoi accepts BSHD Q/K/V tensors plus the packed prefix/video layout and
transformer-layer index. It applies the configured sparse routing policy and
selects the native backend from the input tensor's CUDA device.

## 🔧 Install and build

Install Anemoi into the environment that already contains the model's PyTorch
runtime, then build the extension for the target architecture:

```bash
MPA_SKIP_CUDA_BUILD=1 python -m pip install -e . --no-build-isolation

# RTX 4090 / SM89
MPA_BUILD_COMPONENTS=sm89 scripts/build_attention_cuda.sh

# RTX 5090 / SM120
MPA_BUILD_COMPONENTS=sm120 scripts/build_attention_cuda.sh
```

The architecture-level `sm120` component includes the shared packing/output
assembly operations and the internal SM120 kernels for both Q64 and Q128.
Third-party builds do not need to name internal extension components.
`MPA_PYTHON` selects the Python executable and `MPA_CUDA_HOME` selects the CUDA
toolkit when their defaults are unsuitable. The generic attention API has no
MiniMax-H3 runtime dependency.

## 🌐 Public interface

All public symbols are exported from `anemoi` and
`anemoi.layers.attention`:

```python
import torch

from anemoi import (
    NVFP4Calibration,
    QuantConfig,
    SparseConfig,
    VisualLayout,
    anemoi_attention,
)


def anemoi_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    layout: VisualLayout,
    layer: int,
    sparse_config: SparseConfig | None = None,
    quant_config: QuantConfig | None = None,
    calibration: NVFP4Calibration | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor: ...
```

| Public type | Purpose |
| --- | --- |
| `VisualLayout(video_shape, prefix_tokens=0)` | Describes the packed prefix and 3-D visual token grid |
| `SparseConfig(...)` | Selects Q64/Q128, the dropped-block ratio, and optional SM89/SM120 Mean/MaxPool routing fusion |
| `QuantConfig(...)` | Splits retained visual blocks across public precision phases and selects prefix precision |
| `NVFP4Calibration(q_scale=1.0, k_scale=1.0, v_scale=1.0)` | Optionally supplies independent tensor-level NVFP4 Q/K/V scales |

## 📋 Tensor and layout contract

`query`, `key`, and `value` must:

- share shape `[1, sequence_tokens, heads, head_dim]` in BSHD order
- be physically contiguous CUDA tensors on the same device
- share `torch.float16` or `torch.bfloat16` dtype

`VisualLayout.video_shape` is `(frames, height, width)` in token-grid units.
The sequence must satisfy:

```text
sequence_tokens = prefix_tokens + frames * height * width
```

`prefix_tokens` is a nonnegative count of contiguous tokens before the visual
tokens. Both packed prefix-plus-video and visual-only layouts are valid:

```python
from anemoi import VisualLayout

text_and_video = VisualLayout((37, 24, 42), prefix_tokens=951)
visual_only = VisualLayout((37, 24, 42), prefix_tokens=0)
```

`layer` is a nonnegative, zero-based transformer-layer index. It selects the
first matching `SparseConfig.layer_sparsity_bands` entry; the base
`sparsity_ratio` applies when no band matches.

The stable call is inference self-attention with `attn_mask=None`,
`dropout_p=0.0`, `is_causal=False`, and `scale=None` or
`1 / sqrt(head_dim)`. SM89 accepts head dimensions 64 and 128; SM120 currently
accepts head dimension 128. The
result has the same BSHD shape and dtype as `query`.

## ⚙️ Sparse and precision configuration

`SparseConfig.query_block_size` is `64` or `128`. `sparsity_ratio` and every
per-layer sparsity value must be finite and in `[0, 1)`; each value is the
fraction of visual block pairs dropped. Layer bands are sorted, disjoint
`(first, last, sparsity)` tuples with zero-based, half-open `[first, last)`
ranges. The default is Q64 with `sparsity_ratio=0.80` and
`layer_sparsity_bands=()`, so every sparse layer drops 80% of routed visual
block pairs. Explicit layer bands remain available as overrides.

`SparseConfig.maxpool_weight` is finite in `[0, 1]` and defaults to `0`. On
SM89 and SM120 Q64/Q128, let `w = maxpool_weight`; the independently normalized
routing probability is

```text
(1 - w) * softmax(Q_mean @ K_mean^T / sqrt(head_dim))
    + w * softmax(Q_max @ K_max^T / sqrt(head_dim))
```

Mean and elementwise max descriptors use the same ragged Q/K block traversal;
invalid padding tokens do not participate in the maximum. Weight `0` preserves
the legacy mean-only path without max-pool allocation, weight `1` runs only the
max probability map, and an interior weight computes and fuses both maps.
SM89 and SM120 both fuse max descriptors into their existing donor-first Q/K
preparation traversal. Thus the disabled path performs no max allocation or
max reduction, while an enabled path adds no second raw-Q/K traversal.

```python
from anemoi import SparseConfig

q64 = SparseConfig()
q128 = SparseConfig(query_block_size=128)
maxpool = SparseConfig(query_block_size=128, maxpool_weight=0.5)
custom = SparseConfig(
    query_block_size=64,
    sparsity_ratio=0.80,
    layer_sparsity_bands=((2, 4, 0.88), (4, 5, 0.78)),
)
```

`QuantConfig.nvfp4_ratio`, `int8_ratio`, and `fp16_ratio` divide the retained
visual blocks. They must be finite, nonnegative, and sum to one within
`1e-6`; a positive ratio activates that precision phase.

```python
from anemoi import QuantConfig

mixed = QuantConfig(
    nvfp4_ratio=0.60,
    int8_ratio=0.25,
    fp16_ratio=0.15,
    prefix_kv_precision="int8",
    prefix_query_precision="int8",
)
```

The generic API applies sparse attention on every call. Dense-first request or
layer scheduling, when desired, remains the model adapter's responsibility.

## 📊 Stable capability matrix

SM120 supports both visual-only and packed prefix-plus-video layouts for every
cell below. A cell name means each named retained-block ratio is positive and
all other public precision ratios are zero.

| Retained visual phases | SM120 Q64 | SM120 Q128 | Prefix K/V | Prefix queries |
| --- | --- | --- | --- | --- |
| NVFP4 | Stable | Stable | NVFP4, INT8, or FP16 | FP16 |
| INT8 | Stable | Stable | NVFP4, INT8, or FP16 | INT8 or FP16 |
| FP16 | Stable | Stable | NVFP4, INT8, or FP16 | FP16 |
| NVFP4 + INT8 | Stable | Stable | NVFP4, INT8, or FP16 | INT8 or FP16 |
| NVFP4 + FP16 | Stable | Stable | NVFP4, INT8, or FP16 | FP16 |
| INT8 + FP16 | Stable | Stable | NVFP4, INT8, or FP16 | INT8 or FP16 |
| NVFP4 + INT8 + FP16 | Stable | Stable | NVFP4, INT8, or FP16 | INT8 or FP16 |

Thus the stable SM120 Q128 set is a superset of the Q64 set; at present the
two public sets are equal. Both query geometries support the optional
`maxpool_weight` routing blend described above. For `prefix_tokens > 0`,
prefix-query INT8 requires
an active retained-video INT8 phase. Prefix-query FP16 uses dense
original-dtype attention, while `prefix_kv_precision` independently selects
NVFP4, INT8, or FP16 for prefix K/V. The two prefix fields do not affect
computation when `prefix_tokens=0`.

SM89 supports Q64 and Q128 with pure INT8, pure FP16, or ordered INT8 + FP16
retained visual blocks. Pure phases dispatch their standalone specialization;
the mixed kernel always executes INT8 before FP16. Both prefix fields accept
INT8 or FP16, both query geometries support the same optional
`maxpool_weight` routing blend, and both layout forms are supported. Runtime dispatch requires
an exact SM89 or SM120 CUDA capability.

## 🛠️ NVFP4 calibration

NVFP4 arithmetic uses one tensor-level global scale for each of Q, K, and V.
Omitting `calibration` uses `(1.0, 1.0, 1.0)`. Applications may instead pass
independent finite, positive scales with `NVFP4Calibration`:

Each value becomes an internal CUDA FP32 tensor scale in the dequantization
direction, not an input multiplier, observed `amax`, or inverse scale. For each
16-value group, preparation uses saturating E4M3 group-scale encoding and E2M1
data encoding:

```text
raw_group_scale = (group_amax / 6) / tensor_scale
dequant_scale = E4M3(raw_group_scale) * tensor_scale
E2M1_data = E2M1(value / dequant_scale)
```

A model provider may derive each tensor scale as
`observed_tensor_amax * margin / (6 * 448)`, with the margin selected for that
model and calibration workload.

```python
from anemoi import (
    NVFP4Calibration,
    QuantConfig,
    SparseConfig,
    VisualLayout,
    anemoi_attention,
)

layout = VisualLayout(video_shape, prefix_tokens=prefix_tokens)
sparse = SparseConfig(query_block_size=128)
nvfp4 = QuantConfig(
    nvfp4_ratio=1.0,
    int8_ratio=0.0,
    fp16_ratio=0.0,
    prefix_kv_precision="nvfp4",
    prefix_query_precision="fp16",
)

# Generic/default NVFP4 uses unity Q/K/V scales.
unity_output = anemoi_attention(
    query,
    key,
    value,
    layout=layout,
    layer=layer_index,
    sparse_config=sparse,
    quant_config=nvfp4,
)

# Illustrative only; these values are not calibrated for any model.
calibrated_output = anemoi_attention(
    query,
    key,
    value,
    layout=layout,
    layer=layer_index,
    sparse_config=sparse,
    quant_config=nvfp4,
    calibration=NVFP4Calibration(
        q_scale=0.010,
        k_scale=0.012,
        v_scale=0.020,
    ),
)
```

The executor caches the device scalar tensors internally; callers manage only
the three Python scale values. Calibration is used whenever NVFP4 is active
for retained video blocks or prefix K/V.

MiniMax-H3 is one user of this generic contract. Its model adapter explicitly
loads the exact per-layer Q/K/V calibration owned by the MiniMax-H3 package and
passes the selected layer's `NVFP4Calibration` to `anemoi_attention`. The
generic attention package neither reads nor depends on that model resource.

## 🔄 Model integration

Create immutable layout/config objects at model setup and replace a compatible
FlashAttention-style call at the point where BSHD Q/K/V are available:

```python
from anemoi import QuantConfig, SparseConfig, VisualLayout, anemoi_attention

layout = VisualLayout(video_shape, prefix_tokens=prefix_tokens)
sparse_config = SparseConfig(query_block_size=64)
quant_config = QuantConfig(int8_ratio=1.0)

# Before
output = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)

# After
output = anemoi_attention(
    query,
    key,
    value,
    layout=layout,
    layer=layer_index,
    sparse_config=sparse_config,
    quant_config=quant_config,
)
```

The caller supplies the current layout and zero-based layer index; no
MiniMax-H3 runner, layout observer, or model adapter is required.

## ⚠️ Validation boundary

The API validates layout/config metadata and fixed attention semantics before
dispatch. Tensor shape, dtype, dense layout, device placement, native-extension
availability, GPU capability, and architecture-specific precision cells are
also checked before or at the native boundary. Invalid metadata or unsupported
FlashAttention options raise `TypeError` or `ValueError`; missing native
capability or an incompatible architecture/precision selection raises
`RuntimeError`.

Every SM120 cell listed above is admitted by the production dispatcher. Values
outside the documented matrix are not part of the stable public contract.
