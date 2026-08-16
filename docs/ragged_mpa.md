# Ragged 2-D mixed-precision attention

## Partition contract

For every positive latent grid `H x W`, the production `stripe-compact`
partition creates exactly `ceil(H * W / 64)` connected blocks. Every real
token appears exactly once, no block contains more than 64 tokens, and padding
lanes never become logical tokens. This differs from the historical regular
`compact-grid` strategy. For example, `24x42` needs 16 ragged blocks instead
of 18 regular blocks, increasing useful physical QK work from 76.56% to 96.90%.

The cached host-side dynamic program cuts complete row bands. Each band follows
a column-serpentine connected path and is divided into balanced intervals. It
evaluates both grid orientations, then lexicographically minimizes perimeter,
bounding-box waste, spatial moment, and aspect error while preserving the
capacity-minimum block count.

Only intervals inside one band are balanced. Blocks from different bands can
have different token counts, and the global difference is not constrained to
one token.

## Exact denominators and masks

The request-static device layout stores:

- `indices`: packed-slot to original-video-token mapping;
- `slot_valid`: whether a physical lane contains a real token;
- `counts`: exact real-token count for every block;
- `inverse`: original-video-token to packed-slot mapping;
- `anchors`: self and shared-boundary adjacency within each frame.

The native single-load packer gathers Q/K/V through `indices` and writes zero
for invalid lanes. DraftMap Q/K means divide by each block's true `counts`, not
by 64. Optional diagonal-Jensen second moments use the same denominator.
Attention receives exact key counts for prefix and video blocks, so invalid
K64 columns are masked by the existing native kernel. Output assembly writes
only real packed video slots through `inverse`.

## Routing and precision

The routing score is the row-softmax probability of pooled QK logits. Setting
`diag_jensen: true` adds the diagonal second-moment correction without changing
partition or budget semantics. Routing performs an exact global top-k over all
video block pairs for each batch/head.
Same-frame self and shared-boundary neighbours are mandatory FP16 anchors, but
they consume the retained budget by borrowing FP8 seats when needed. If a very
small grid requests fewer retained pairs than there are mandatory anchors, the
router raises retention only to that minimum feasible anchor count instead of
rejecting the resolution.

The released schedule drops 88%, 82%, and 58% of video block pairs across the
three sparse layer regions. Retained pairs use FP8/FP16=80/20 before anchor
promotion. Prefix K/V remains exact FP16, scheduled dense calls and prefix
query rows use original-dtype framework SDPA, and skipped pairs receive no
compensation.

The current integrated executor is the optimized native SM89 Q64xK64 kernel.
