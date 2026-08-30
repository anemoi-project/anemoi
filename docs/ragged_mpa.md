# Ragged 2-D mixed-precision attention

## Partition contract

For every positive latent grid `H x W` and physical block capacity `C`, the
production compact partition creates the minimum
`B = ceil(H * W / C)` connected blocks. With
`q, r = divmod(H * W, B)`, exactly `r` blocks contain `q + 1` real tokens and
the remaining blocks contain `q`. Every real token appears exactly once,
padding lanes never become logical tokens, and global block-mass spread is
therefore at most one. This differs from the historical regular `compact-grid`
strategy. For example, `24x42/C64` needs 16 ragged blocks instead of 18 regular
blocks, increasing useful physical QK work from 76.56% to 96.90%.

The cached host-side search first targets a near-square block lattice in both
grid orientations. It cuts ragged outer bands with exact aggregate mass, walks
each band on the perpendicular axis, and tries deterministic end-loaded and
even placements of the required `q+1` blocks. The resulting one-cell seams
absorb non-divisible row totals while keeping individual blocks compact. The
legacy complete-band DP remains a candidate, so the selected partition cannot
regress its lexicographic objective: perimeter, bounding-box waste, spatial
moment, then aspect error.

## Exact denominators and masks

The request-static device layout stores:

- `indices`: packed-slot to original-video-token mapping;
- `slot_valid`: whether a physical lane contains a real token;
- `counts`: exact real-token count for every block;
- `inverse`: original-video-token to packed-slot mapping.

The native single-load packer gathers Q/K/V through `indices` and writes zero
for invalid lanes. DraftMap Q/K means divide by each block's true `counts`, not
by 64. Attention receives exact key counts for prefix and video blocks, so
invalid K64 columns are masked by the existing native kernel. Output assembly
writes only real packed video slots through `inverse`.

## Routing and precision

The production routing score is the row-softmax probability of pooled QK
logits, followed by exact global top-k per batch/head. The default applies 80%
sparsity (20% keep) uniformly to every sparse layer; optional layer bands can
override selected half-open layer ranges. Retained pairs, prefix K/V, and prefix
query rows use the native INT8 phase by default. Scheduled-dense calls use
original-dtype framework SDPA, and unselected pairs are dropped directly.

The integrated executors are the optimized native SM89 Q64/Q128 and SM120 Q64/Q128
paths; runtime device capability and `query_block_size` select the kernel.
