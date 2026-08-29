# Ragged 2-D mixed-precision attention

## Partition contract

For every positive latent grid `H x W` and physical block capacity `C`, the
production `stripe-compact` partition creates the minimum
`B = ceil(H * W / C)` connected blocks. With
`q, r = divmod(H * W, B)`, exactly `r` blocks contain `q + 1` real tokens and
the remaining blocks contain `q`. Every real token appears exactly once,
padding lanes never become logical tokens, and global block-mass spread is
therefore at most one. This differs from the historical regular `compact-grid`
strategy. For example, `24x42/C64` needs 16 ragged blocks instead of 18 regular
blocks, increasing useful physical QK work from 76.56% to 96.90%.

The cached host-side dynamic program cuts complete row bands. Each band follows
a column-serpentine connected path and is divided into intervals of the global
`q/q+1` sizes. Bands that cannot satisfy those sizes are infeasible. Among the
remaining candidates the DP evaluates both grid orientations, then
lexicographically minimizes perimeter, bounding-box waste, spatial moment, and
aspect error. A centered internal-band-boundary tie makes equivalent stripe
orders deterministic without overriding mass balance or geometry.

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
logits, followed by exact global top-k per batch/head. The calibrated Mean20
L2-L49 schedule averages 80% sparsity (20% keep) over sparse layers. Retained
pairs, prefix K/V, and prefix query rows use the native INT8 phase by default.
Scheduled-dense calls use original-dtype framework SDPA, and unselected pairs
are dropped directly.

The integrated executors are the optimized native SM89 Q64 and SM120 Q64/Q128
paths; runtime device capability and `query_block_size` select the kernel.
