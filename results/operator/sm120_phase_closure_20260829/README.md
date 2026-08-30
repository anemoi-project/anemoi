# SM120 phase-closure audit

This development ledger compares the frozen `5542126` baseline with retained
closure commit `4b31468`, built with CUDA 13.1 for `sm_120a`.  It audits native
specializations, not public API promotion.  The persistent mathematical state
is `ro/m/d`; all phase bodies are lexical scopes in the shared composer and
shared memory is `max(low, high, output)`, never their sum.

## Artifacts

| Artifact | Path | SHA256 |
|---|---|---|
| Build log | `/tmp/anemoi-sm120-phase-closure-baseline-build.log` | `9a4bc3e9fa4f23a041e7ddc1faef8b92323acffefd59d078d929adca83a8a575` |
| Extension | `/tmp/anemoi-sm120-phase-closure-baseline.so` | `817fe2523fce0fcefd3cee07fc40454b155053e43fb68cacee4a041c4af42683` |
| Resource JSON | `/tmp/anemoi-sm120-phase-closure-baseline.resources.json` | `cb2b65dc30b17c3ee8a39aed49311d905ba48a1469c9bf001d656940397be42c` |
| Line-info SASS | `/tmp/anemoi-sm120-phase-closure-baseline.sass` | `d802723bd87df27f823dfe69f35725b168fbed6759c4de3aa7b490a072ac8ea2` |
| Final build log | `/tmp/anemoi-sm120-phase-closure-accepted-build.log` | `87a71216396706672c507fdc075c26f80e5ed7dbf9abe29a823fa4876097feff` |
| Final extension | `/tmp/anemoi-sm120-phase-closure-accepted.so` | `08ae61320b5788b807efab5e4a13afac61d04aa54f777f85f782662c9b3755bc` |
| Generated-SASS gate | `/tmp/anemoi-sm120-phase-closure-accepted-gate.json` | `e959fd7a0bb1dda744c1ecceb63c9c389108656c26d3c98f0eea80ec9500b7f1` |

All kernels use 128 threads.  Q64 uses 25,088 bytes of dynamic shared memory
without FP16 and 32,768 bytes with FP16; Q128 uses 33,280 and 49,152 bytes.
The register/shared-memory limits permit 3 active Q64 CTAs or 2 active Q128
CTAs per SM for every row below.

## Complete native matrix

`Resource` is `registers / stack / spill-store / spill-load`, in bytes except
registers.  `Pure max` is the component-wise maximum of the homogeneous
phases.  Every row passes the source-scope check: only `ro/m/d` is declared
outside the phase scopes.  That check is necessary but is not proof of
generated-code liveness.

| Q | Runtime phases | Native label | Resource | Pure max | SASS attribution | Historical coverage |
|---:|---|---|---:|---:|---|---|
| 64 | FP16 | `q64_fp16` | `168/0/0/0` | same | phase-local, no traffic | control |
| 64 | INT8 | `q64_int8` | `168/0/0/0` | same | phase-local, no traffic | control |
| 64 | INT8 -> FP16 | `q64_int8_fp16` | `168/24/36/32` | `168/0/0/0` | **entry metadata crosses boundary**; recurring slots 0/4 are current peak | Q64 boundary/launch-bound work only |
| 64 | MXFP8 | `q64_mxfp8` | `166/0/0/0` | same | phase-local, no traffic | control |
| 64 | MXFP8 -> FP16 | `q64_mxfp8_fp16` | `168/0/0/0` | `168/0/0/0` | no traffic | FP16-count reload only |
| 64 | NVFP4 | `q64_nvfp4` | `167/0/0/0` | same | phase-local, no traffic | control |
| 64 | NVFP4 -> FP16 | `q64_nvfp4_fp16` | `168/0/0/0` | `168/0/0/0` | no traffic | FP16-count reload only |
| 64 | NVFP4 -> INT8 | `q64_nvfp4_int8` | `168/8/8/16` | `168/0/0/0` | one 64-bit NVFP4 copy-helper slot; stored/reloaded entirely inside NVFP4, not across the INT8 boundary | no direct candidate |
| 64 | NVFP4 -> INT8 -> FP16 | `q64_nvfp4_int8_fp16` | `168/0/0/0` | `168/0/0/0` | no traffic | retained Q128 remedies plus Q64 regression checks |
| 64 | NVFP4 -> MXFP8 | `q64_nvfp4_mxfp8` | `168/0/0/0` | `168/0/0/0` | no traffic; diagnostic/pending route | negative control |
| 64 | NVFP4 -> MXFP8 -> FP16 | `q64_nvfp4_mxfp8_fp16` | `168/0/0/0` | `168/0/0/0` | no traffic | candidate-15 regression control |
| 128 | FP16 | `q128_fp16` | `255/16/12/12` | same | homogeneous FP16 peak | control |
| 128 | INT8 | `q128_int8` | `255/8/8/8` | same | homogeneous INT8 pointer peak | control |
| 128 | INT8 -> FP16 | `q128_int8_fp16` | `255/40/96/80` | `255/16/12/12` | **entry metadata crosses boundary**; remaining slots are persistent state/current-phase peaks | not covered by retained triple-only offset rule |
| 128 | MXFP8 | `q128_mxfp8` | `254/0/0/0` | same | phase-local, no traffic | control |
| 128 | MXFP8 -> FP16 | `q128_mxfp8_fp16` | `255/40/92/68` | `255/16/12/12` | recurring `d`/score accumulation and FP16 peak; no stable future-metadata slot demonstrated | FP16-count reload; candidate-15 negative control |
| 128 | NVFP4 | `q128_nvfp4` | `255/0/0/0` | same | phase-local, no traffic | control |
| 128 | NVFP4 -> FP16 | `q128_nvfp4_fp16` | `255/24/72/48` | `255/16/12/12` | scaled persistent `ro` transition plus FP16 peak | FP16-count and NV localization candidates |
| 128 | NVFP4 -> INT8 | `q128_nvfp4_int8` | `255/16/20/16` | `255/8/8/8` | recurring INT8 current-phase invariant/peak, no future phase | candidate-07 controls |
| 128 | NVFP4 -> INT8 -> FP16 | `q128_nvfp4_int8_fp16` | `255/8/8/8` | `255/16/12/12` | one recurring INT8 current-phase 64-bit victim; no boundary metadata slot | candidates 01--15, candidate 07 retained |
| 128 | NVFP4 -> MXFP8 | `q128_nvfp4_mxfp8` | `255/0/0/0` | `255/0/0/0` | no traffic | candidate-15 control |
| 128 | NVFP4 -> MXFP8 -> FP16 | `q128_nvfp4_mxfp8_fp16` | `255/32/92/76` | `255/16/12/12` | persistent online-state/current MXFP8/FP16 peak; no stable future-metadata slot demonstrated | FP16-count reload; candidate-15 rejected |

## Proven boundary defect

Line-info SASS gives the same three non-mathematical values in both plain
INT8 -> FP16 kernels:

| Value | Q64 stack slot | Q128 stack slot | Definition | First FP16 reload |
|---|---:|---:|---|---|
| `route_low_iterations` | `+0x8` | `+0x18` | entry `fp8_count[metadata_row]` | FP16 LUT indexing |
| `kv_head` | `+0xc` | `+0x20` | entry `head_id / num_kv_groups` | FP16 K/V addressing |
| `num_kv_heads` | `+0x10` | `+0x1c` | entry `num_qo_heads / num_kv_groups` | FP16 K/V addressing |

These are the first production paths that fail the strict phase-closure
constraint.  The existing FP16-count boundary reload is already correct and
must not be generalized blindly: historical candidate 15 reloaded the middle
count and worsened Q128 INT8/MXFP8 and Q64 triple resources.  The next
candidate therefore targets only the three SASS-proven values above.

## Retained strict closure

Commit `4b31468` closes the two plain INT8 -> FP16 defects at their shared
root.  The FP16 phase reloads its route count at the boundary and derives its
Q/K/V head rows from fresh CTA/grid special-register reads there.  The entry
IDs make a volatile shared-memory round trip before INT8 starts; this prevents
ptxas from value-numbering the entry IDs into the later FP16 reads without
creating a phase-to-phase hand-off.  The derived values remain FP16-local.

| Native label | Baseline | Retained | Delta |
|---|---:|---:|---:|
| `q64_int8_fp16` | `168/24/36/32` | `168/16/32/16` | `0/-8/-4/-16` |
| `q128_int8_fp16` | `255/40/96/80` | `255/32/40/56` | `0/-8/-56/-24` |

The other 20 compiled symbols are resource-identical to the frozen baseline.
The generated-SASS gate requires a second `CTAID.Z` and `CTAID.Y` read before
the first FP16 Q address for both Q64 and Q128; both final symbols pass with
two reads per axis.  Line-info SASS confirms that remaining local-memory
traffic is contained within one phase or holds permitted online state; the
three cross-boundary metadata slots above are gone.

Two smaller-looking candidates were not retained:

- Reloading only the route count removed more spill but left both KV metadata
  values crossing the boundary, so it did not satisfy the phase contract.
- Re-reading CTA/grid separately for Q and KV satisfied the contract but added
  redundant special-register work.  Reusing the FP16-phase Q head values for
  one combined KV row removed that duplication.
- Packing both entry IDs into one 64-bit shared round trip looked cheaper, but
  ptxas increased Q64/Q128 spill and again removed the fresh `CTAID.Z` read;
  the generated-SASS gate rejected it.

## Complete closure decision

Every accepted row now meets the original constraint: only `ro/m/d` survives
between phase bodies.  Local traffic named below is a single-phase peak, not a
phase hand-off.

| Q | Path family | Final boundary classification |
|---:|---|---|
| 64 | FP16, INT8, MXFP8, NVFP4 | homogeneous controls; no phase boundary |
| 64 | INT8 -> FP16 | fixed; future route and KV metadata rematerialized in FP16 |
| 64 | MXFP8 -> FP16, NVFP4 -> FP16 | no cross-boundary local traffic |
| 64 | NVFP4 -> INT8 | one NVFP4 copy-helper slot, defined and consumed inside NVFP4 |
| 64 | NVFP4 -> INT8 -> FP16 | no cross-boundary local traffic |
| 64 | NVFP4 -> MXFP8 -> FP16 | no cross-boundary local traffic |
| 128 | FP16, INT8, MXFP8, NVFP4 | homogeneous controls; no phase boundary |
| 128 | INT8 -> FP16 | fixed; future route and KV metadata rematerialized in FP16 |
| 128 | MXFP8 -> FP16 | `d`/score and FP16 current-phase peaks only |
| 128 | NVFP4 -> FP16 | permitted scaled `ro` hand-off plus FP16 current-phase peak |
| 128 | NVFP4 -> INT8 | INT8 current-phase invariant/peak only |
| 128 | NVFP4 -> INT8 -> FP16 | one recurring INT8 current-phase victim only |
| 128 | NVFP4 -> MXFP8 | no local traffic |
| 128 | NVFP4 -> MXFP8 -> FP16 | permitted online state plus MXFP8/FP16 current-phase peaks |

Q64 NVFP4 -> MXFP8 without FP16 remains a compile/correctness diagnostic cell,
not an accepted runtime configuration.  It is spill-free and did not expose a
closure defect, but this audit does not promote it.

## Real-capture validation

`benchmark.json` records all 21 accepted cells on the 5 s Dense capture.  All
use matched per-Q routes with measured effective sparsity `0.7999994293` (Q64) or
`0.8000022827` (Q128); the same-workload Dense latency is `50.0422 ms`.
Numerical metrics use that Dense output only.  Maxpool and anchors are disabled.

The retained plain INT8 -> FP16 change was additionally tested against the
frozen binary in balanced forward/reverse process order.  On the 10 s capture,
Q64 changed from `23.6614` to `23.7514 ms` (`+0.38%`) and Q128 from `21.4198`
to `21.6883 ms` (`+1.25%`).  Relative to identical pure-INT8 controls, the
estimated closure cost is `+0.72%` for Q64 and `+0.51%` for Q128.  Exact
retained/total blocks are `265421/1327104` and `66355/331776`; anchors and
maxpool are disabled.  A separate 5 s replay reproduced Q64/Q128 output and
LSE bitwise.  Dense remains the only accuracy reference.

Final resource and runtime records are `resources.json` and `benchmark.json`.
