# C++ and CUDA Extensions

This directory contains the native SM89 and SM120 mixed-attention kernels used
by `anemoi.layers.attention.mpa`, including fused operand preparation and
DraftMap scoring/routing. Build the native executor with
`scripts/build_attention_cuda.sh`.
