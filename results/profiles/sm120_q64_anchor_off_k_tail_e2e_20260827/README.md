# SM120 Q64 K-tail development profile

This directory preserves the opt-in K-tail layer gates and the four-case
quality/timing summary from
`private/feature/sm120-k-tail-r2-layer-gate-20260827`.

The YAML is an experiment profile, not the production default. It uses Mean
outside seven half-open layer bands, K-tail r1 on six selected layers, K-tail
r2 on layer 21, and disables Anchors for the recorded comparison. The canonical
SM89/SM120 profiles omit the two experimental K-tail fields and resolve to
Mean with no layer overrides; Anchors remain enabled. Copy this profile before
changing the layer schedule or Anchor policy.

The JSON compares K-tail and Mean only against the matching Dense output for
each prompt. Across the four recorded cases, K-tail's aggregate transformer
time was effectively unchanged relative to Mean while its quality effect was
prompt-dependent; it is therefore exposed as a configuration-controlled SM120
Q64 path rather than promoted globally.
