#pragma once

#include <torch/extension.h>

#include <optional>
#include <tuple>

// Native Q64 x K64 FP16 attention used by both contiguous-1D and compact
// aligned-8x8 layouts. block_ids are absolute K64 block indices; every
// physical block has an explicit valid-token count so partial compact blocks
// are masked in both score and value accumulation.
std::tuple<torch::Tensor, torch::Tensor> k64_fp16_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_ids,
    torch::Tensor block_counts,
    torch::Tensor valid_k_counts,
    double softmax_scale);

// Native Q64xK64 mixed executor. Q/K use the project INT8 tensor-core
// representation and V uses E4M3; selected rescue blocks consume the exact
// FP16 operands. The two list tensors must alias one compact absolute-stage
// list: [FP8 video stages][FP16 video stages][unused]. Exact prefix stages are
// implicit in the FP16 count and do not occupy list slots.
std::tuple<torch::Tensor, torch::Tensor> k64_mixed_attention_forward(
    torch::Tensor q8,
    torch::Tensor k8,
    torch::Tensor v8,
    torch::Tensor q16,
    torch::Tensor k16,
    torch::Tensor v16,
    torch::Tensor fp8_block_ids,
    torch::Tensor fp8_block_counts,
    torch::Tensor fp16_block_ids,
    torch::Tensor fp16_block_counts,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale);

std::tuple<torch::Tensor, torch::Tensor> q128_k64_mixed_attention_forward(
    torch::Tensor q8,
    torch::Tensor k8,
    torch::Tensor v8,
    torch::Tensor q16,
    torch::Tensor k16,
    torch::Tensor v16,
    torch::Tensor fp8_block_ids,
    torch::Tensor fp8_block_counts,
    torch::Tensor fp16_block_ids,
    torch::Tensor fp16_block_counts,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale);

// Audit-only pure low-precision entry for isolating the inherited
// Sparge/Sage wait_group<1> pipeline on the same absolute Q64xK64 route.
std::tuple<torch::Tensor, torch::Tensor> k64_fp8_attention_forward(
    torch::Tensor q8,
    torch::Tensor k8,
    torch::Tensor v8,
    torch::Tensor block_ids,
    torch::Tensor block_counts,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor valid_k_counts,
    double softmax_scale);

std::tuple<torch::Tensor, torch::Tensor> q128_k64_fp8_attention_forward(
    torch::Tensor q8,
    torch::Tensor k8,
    torch::Tensor v8,
    torch::Tensor block_ids,
    torch::Tensor block_counts,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor valid_k_counts,
    double softmax_scale);

// Integration-private adaptive-2D preparation.  A cached logical-to-physical
// token map drives one fused BF16/FP16 -> FP16 Q/K/V pack into K64 blocks;
// invalid physical lanes are written as exact positive zero.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
pack_indexed_k64_qkv_fp16(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor token_indices,
    torch::Tensor slot_valid);

// Released-H3 specialization of the same preparation boundary.  Prefix K/V
// is copied into leading K64 blocks while indexed video Q/K/V is packed into
// the suffix, avoiding separate prefix padding and K/V concatenation kernels.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
pack_h3_k64_qkv_fp16(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor video_token_indices,
    torch::Tensor video_slot_valid,
    int64_t prefix_tokens);

// One-pass inverse-raster scatter, prefix append, BHSD->BSHD transform, and
// FP16-video conversion to the original H3 input/output dtype.
torch::Tensor assemble_h3_k64_output(
    torch::Tensor prefix_output_bhsd,
    torch::Tensor video_output_bhsd_fp16,
    torch::Tensor video_inverse_indices);

std::tuple<torch::Tensor, torch::Tensor> preprocess_v_fp8(
    torch::Tensor value);

// Quantization-only half of V preprocessing for a caller-provided
// channel-major, 16-token-permuted FP16 [B,Hkv,D,K] tensor.
std::tuple<torch::Tensor, torch::Tensor> quantize_permuted_v_fp8(
    torch::Tensor permuted_value);

// Package-private released-H3 specialization.  Its producer-provided
// per-physical-stage/channel absmax removes the first full read of permuted V.
std::tuple<torch::Tensor, torch::Tensor>
quantize_permuted_v_fp8_h3_vpartials(
    torch::Tensor permuted_value,
    torch::Tensor value_stage_amax);

// Package-private no-permuted-V production consumer.  Consumes the exact-H3
// physical K64-stage packed V [1,14,42624,128] and producer stage/channel
// absmax, transposes/permutates into the inherited FP8 V layout while quantizing.
std::tuple<torch::Tensor, torch::Tensor>
quantize_packed_v_fp8_h3_vpartials(
    torch::Tensor packed_value,
    torch::Tensor value_stage_amax);

// Production raw-video preprocessing boundary.  Converts same-dtype
// FP16/BF16 frame-major raster [B,H,F*Y*X,D] into FP16 logical 8x16 patch
// order [B,H,R*128,D]. BF16 is narrowed before storage and every virtual edge
// slot is exact positive FP16 zero.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
pack_raster_qkv_fp16(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t frames,
    int64_t height,
    int64_t width);

// Candidate/production boundary for the address-mapped Q/K path.  Only V is
// physically materialized in logical 8x16 patch order; raw Q/K stay in their
// caller-owned frame-major raster layout.
torch::Tensor pack_raster_v_fp16(
    torch::Tensor value,
    int64_t frames,
    int64_t height,
    int64_t width);

// Experimental phase-aware boundary: K/V are materialized together while Q
// remains caller-owned raw raster FP16.  This reuses the production fused KV
// copy rather than launching two independent pack kernels.
std::tuple<torch::Tensor, torch::Tensor> pack_raster_kv_fp16(
    torch::Tensor key,
    torch::Tensor value,
    int64_t frames,
    int64_t height,
    int64_t width);

// Raster-aware derivative of the inherited Q128/K64 INT8 quantization.
// key_mean is source-bound FP16 [B,Hkv,D] when smooth_k=true and a typed empty
// tensor otherwise.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
quantize_packed_raster_qk_int8(
    torch::Tensor packed_query,
    torch::Tensor packed_key,
    torch::Tensor key_mean,
    int64_t frames,
    int64_t height,
    int64_t width,
    bool smooth_k);

// Native INT4 path: symmetric signed values [-7, 7] are packed as two
// two's-complement nibbles per byte for the S4 Tensor Core mainloop.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
quantize_packed_raster_qk_int4(
    torch::Tensor packed_query,
    torch::Tensor packed_key,
    torch::Tensor key_mean,
    int64_t frames,
    int64_t height,
    int64_t width,
    bool smooth_k);

// Fused address-map + quantization path.  The two reduction passes gather the
// required raster block directly from raw Q/K, while only the final INT8
// operands are written in logical Q128 / physical K64 order.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
quantize_raster_qk_int8(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor key_mean,
    int64_t frames,
    int64_t height,
    int64_t width,
    bool smooth_k);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
quantize_raster_qk_int4(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor key_mean,
    int64_t frames,
    int64_t height,
    int64_t width,
    bool smooth_k);

// Fused mixed-phase preparation. The first raw Q/K read materializes FP16
// rescue operands and contributes to the shared per-block absmax. The second
// raw read emits compile-time-selected INT4, INT8, or both; INT4 is packed as
// four nibble pairs per 32-bit store. K and V share the same K64 CTA and also
// produce V's channel-major/permuted quantization input.
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
pack_quantize_raster_qkv_low(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_mean,
    int64_t frames,
    int64_t height,
    int64_t width,
    bool smooth_k,
    bool has_int4,
    bool has_int8);

// Additive exact-H3 preparation ABI.  The first twelve tensors have exactly
// the same order and meaning as pack_quantize_raster_qkv_low; the final FP32
// tensor is [1,14,666,128] per-physical-stage/channel V absmax workspace.
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
pack_quantize_raster_qkv_low_h3_vpartials(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_mean);

// Package-private no-permuted-V exact-H3 producer ABI.  The tuple is:
// packed Q/K/V, Q4/scale, K4/scale, Q8/scale, K8/scale, V stage absmax.
// It deliberately neither allocates nor writes full channel-major FP16 V.
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
pack_quantize_raster_qkv_low_h3_packed_vpartials(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_mean);

// Compatibility wrapper for the original INT8-only fused internal ABI.
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
pack_quantize_raster_qkv_int8(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_mean,
    int64_t frames,
    int64_t height,
    int64_t width,
    bool smooth_k);

// Production one-kernel family. packed_ids is one logical backing per query
// row: [INT4 IDs][INT8 IDs][FP16 IDs][zero tail]. O is written in original
// raster-token order. A shaped lse_raw_fp32 requests natural-log FP32 LSE;
// an empty CUDA FP32 tensor is the no-LSE sentinel.
std::tuple<torch::Tensor, torch::Tensor>
packed_raster_mixed_attention_forward(
    torch::Tensor q4,
    torch::Tensor k4,
    torch::Tensor q8,
    torch::Tensor k8,
    torch::Tensor v8,
    torch::Tensor packed_q_fp16,
    torch::Tensor packed_k_fp16,
    torch::Tensor packed_v_fp16,
    torch::Tensor k_mean_fp16,
    torch::Tensor packed_text_k_fp16,
    torch::Tensor packed_text_v_fp16,
    torch::Tensor valid_text_counts,
    torch::Tensor output_raw_fp16,
    torch::Tensor lse_raw_fp32,
    torch::Tensor packed_ids,
    torch::Tensor fp8_counts,
    torch::Tensor fp16_counts,
    torch::Tensor int4_counts,
    torch::Tensor q4_scale,
    torch::Tensor k4_scale,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    int64_t frames,
    int64_t height,
    int64_t width,
    double softmax_scale,
    bool smooth_k,
    bool has_int4,
    bool has_fp8,
    bool has_fp16);

// Final Anemoi-facing output assembly.  The video partition arrives in the
// native attention ABI's contiguous BHSD layout while both dense partitions
// use contiguous BSHD.  The two visual key partitions are normalized with
// their FP32 natural-log LSE states, then the text partition is copied and an
// optional CUDA bool [B,S_text] mask writes exact positive BF16 zero.  An
// empty CUDA bool tensor is the no-mask sentinel.
torch::Tensor assemble_video_text_output(
    torch::Tensor video_output_bhsd,
    torch::Tensor video_lse_bhs,
    torch::Tensor visual_text_output_bshd,
    torch::Tensor visual_text_lse_bhs,
    torch::Tensor text_output_bshd,
    torch::Tensor text_mask);

// Final layout join after visual queries have already traversed both
// video and dense-text K/V in one CTA.  No LSE merge is performed here.
torch::Tensor assemble_fused_visual_text_output(
    torch::Tensor visual_output_bhsd,
    torch::Tensor text_output_bshd,
    torch::Tensor text_mask);

// Test-only probes for the reusable packed-list/raster address contract.
// Neither entry launches attention or claims that the contiguous loaders are
// raster-aware.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
probe_raster8x16_layout(
    torch::Tensor anchor,
    int64_t frames,
    int64_t height,
    int64_t width);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
probe_packed_phase_stages(
    torch::Tensor packed_ids,
    torch::Tensor fp8_counts,
    torch::Tensor fp16_counts,
    torch::Tensor fp4_counts,
    int64_t phase);

torch::Tensor contiguous_mixed_attention_forward(
    torch::Tensor q8,
    torch::Tensor k8,
    torch::Tensor v8,
    torch::Tensor q16,
    torch::Tensor k16,
    torch::Tensor v16,
    torch::Tensor k_mean,
    torch::Tensor output,
    torch::Tensor fp8_lut,
    torch::Tensor fp8_count,
    torch::Tensor fp16_lut,
    torch::Tensor fp16_count,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    double softmax_scale,
    bool smooth_k,
    bool has_fp8,
    bool has_fp16);
