#pragma once

#include <torch/extension.h>

#include <optional>
#include <tuple>
#include <vector>

using H3SM120Prepared = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor>;

H3SM120Prepared prepare_h3_sm120_operands(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor video_token_indices,
    torch::Tensor video_slot_valid,
    torch::Tensor video_valid_counts,
    int64_t prefix_tokens,
    int64_t query_block_size,
    bool has_nvfp4,
    bool has_int8,
    bool has_mxfp8,
    bool has_fp16,
    bool has_prefix_query_int8,
    bool has_maxpool,
    std::optional<torch::Tensor> q_global_scale,
    std::optional<torch::Tensor> k_global_scale,
    std::optional<torch::Tensor> v_global_scale);

torch::Tensor sm120_h3_draft_probability(
    torch::Tensor q_pool,
    torch::Tensor k_pool,
    std::optional<torch::Tensor> q_max_pool,
    std::optional<torch::Tensor> k_max_pool,
    double maxpool_weight);

// The same architecture-neutral GEMM/softmax implementation is linked into
// the SM89 extension under an architecture-owned public symbol.
torch::Tensor sm89_h3_draft_probability(
    torch::Tensor q_pool,
    torch::Tensor k_pool,
    std::optional<torch::Tensor> q_max_pool,
    std::optional<torch::Tensor> k_max_pool,
    double maxpool_weight);

torch::Tensor sm120_h3_k_tail_r1_probability(
    torch::Tensor q_pool,
    torch::Tensor k_pool,
    torch::Tensor packed_k,
    torch::Tensor valid_counts,
    int64_t prefix_blocks);

torch::Tensor sm120_h3_k_tail_r2_probability(
    torch::Tensor q_pool,
    torch::Tensor k_pool,
    torch::Tensor packed_k,
    torch::Tensor valid_counts,
    int64_t prefix_blocks);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sm120_h3_route_precision(
    torch::Tensor probability,
    int64_t n16,
    int64_t n8,
    int64_t n4,
    std::optional<torch::Tensor> anchors,
    std::optional<torch::Tensor> anchor_ids,
    int64_t anchor_count);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sm120_h3_materialize_route(
    torch::Tensor logical_ids,
    torch::Tensor low_counts,
    torch::Tensor middle_counts,
    torch::Tensor high_counts,
    int64_t query_block_size,
    int64_t prefix_blocks,
    int64_t prefix_phase,
    bool prefix_first,
    bool has_high);

// Native SM120 Q64xK64 all-FP16 foundation. Every physical K64 block has an
// explicit valid-token count for score and value tail masking.
std::tuple<torch::Tensor, torch::Tensor> sm120_q64_fp16_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_ids,
    torch::Tensor block_counts,
    torch::Tensor valid_k_counts,
    double softmax_scale);

std::tuple<torch::Tensor, torch::Tensor> sm120_q128_fp16_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_ids,
    torch::Tensor block_counts,
    torch::Tensor valid_k_counts,
    double softmax_scale);

std::vector<int64_t> sm120_q64_fp16_kernel_metadata();
std::vector<int64_t> sm120_q64_mxfp8_kernel_metadata();

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor>
prepare_mxfp8(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value);

#define MPA_DECLARE_INT8_FORWARD(name) \
std::tuple<torch::Tensor, torch::Tensor> name( \
    torch::Tensor q8, torch::Tensor k8, torch::Tensor v8, \
    torch::Tensor q16, torch::Tensor k16, torch::Tensor v16, \
    torch::Tensor block_ids, torch::Tensor int8_block_counts, \
    torch::Tensor fp16_block_counts, torch::Tensor q_scale, \
    torch::Tensor k_scale, torch::Tensor v_scale, \
    torch::Tensor valid_k_counts, int64_t fp16_prefix_blocks, \
    double softmax_scale, bool active_fp16 = true)

MPA_DECLARE_INT8_FORWARD(sm120_q64_int8_attention_forward);
MPA_DECLARE_INT8_FORWARD(sm120_q128_int8_attention_forward);
#undef MPA_DECLARE_INT8_FORWARD

#define MPA_DECLARE_PREFIX_INT8_FORWARD(name) \
torch::Tensor name( \
    torch::Tensor q8, torch::Tensor k8, torch::Tensor v8, \
    torch::Tensor q_scale, torch::Tensor k_scale, torch::Tensor v_scale, \
    torch::Tensor valid_k_counts, int64_t prefix_tokens, double softmax_scale)

MPA_DECLARE_PREFIX_INT8_FORWARD(sm120_q64_prefix_int8_attention_forward);
MPA_DECLARE_PREFIX_INT8_FORWARD(sm120_q128_prefix_int8_attention_forward);
#undef MPA_DECLARE_PREFIX_INT8_FORWARD

std::tuple<torch::Tensor, torch::Tensor> sm120_q64_mxfp8_attention_forward(
    torch::Tensor q_mxfp8,
    torch::Tensor q_mxfp8_scale,
    torch::Tensor k_mxfp8,
    torch::Tensor k_mxfp8_scale,
    torch::Tensor v_mxfp8,
    torch::Tensor v_mxfp8_scale,
    torch::Tensor q_fp16,
    torch::Tensor k_fp16,
    torch::Tensor v_fp16,
    torch::Tensor block_ids,
    torch::Tensor mxfp8_block_counts,
    torch::Tensor fp16_block_counts,
    torch::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale,
    bool active_fp16 = true);

std::tuple<torch::Tensor, torch::Tensor> sm120_q128_mxfp8_attention_forward(
    torch::Tensor q_mxfp8,
    torch::Tensor q_mxfp8_scale,
    torch::Tensor k_mxfp8,
    torch::Tensor k_mxfp8_scale,
    torch::Tensor v_mxfp8,
    torch::Tensor v_mxfp8_scale,
    torch::Tensor q_fp16,
    torch::Tensor k_fp16,
    torch::Tensor v_fp16,
    torch::Tensor block_ids,
    torch::Tensor mxfp8_block_counts,
    torch::Tensor fp16_block_counts,
    torch::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale,
    bool active_fp16 = true);

std::tuple<torch::Tensor, torch::Tensor>
sm120_q128_mxfp8_compact_attention_forward(
    torch::Tensor q_mxfp8,
    torch::Tensor q_mxfp8_scale,
    torch::Tensor k_mxfp8,
    torch::Tensor k_mxfp8_scale,
    torch::Tensor v_mxfp8,
    torch::Tensor v_mxfp8_scale,
    torch::Tensor q_fp16,
    torch::Tensor k_fp16,
    torch::Tensor v_fp16,
    torch::Tensor block_ids,
    torch::Tensor mxfp8_block_counts,
    torch::Tensor fp16_block_counts,
    torch::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale,
    bool active_fp16 = false);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor>
prepare_q64_nvfp4(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_global_scale,
    torch::Tensor k_global_scale,
    torch::Tensor v_global_scale);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor>
prepare_q128_nvfp4(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_global_scale,
    torch::Tensor k_global_scale,
    torch::Tensor v_global_scale);

#define MPA_DECLARE_NVFP4_FORWARD(name) \
std::tuple<torch::Tensor, torch::Tensor> name( \
    torch::Tensor q_nvfp4, torch::Tensor q_nvfp4_scale, \
    torch::Tensor k_nvfp4, torch::Tensor k_nvfp4_scale, \
    torch::Tensor v_nvfp4, torch::Tensor v_nvfp4_scale, \
    torch::Tensor q_fp16, torch::Tensor k_fp16, torch::Tensor v_fp16, \
    torch::Tensor block_ids, torch::Tensor nvfp4_block_counts, \
    torch::Tensor fp16_block_counts, torch::Tensor valid_k_counts, \
    torch::Tensor q_global_scale, torch::Tensor k_global_scale, \
    torch::Tensor v_global_scale, \
    int64_t fp16_prefix_blocks, \
    double softmax_scale, bool active_fp16 = true)

MPA_DECLARE_NVFP4_FORWARD(sm120_q64_nvfp4_attention_forward);
MPA_DECLARE_NVFP4_FORWARD(sm120_q128_nvfp4_attention_forward);
#undef MPA_DECLARE_NVFP4_FORWARD

#define MPA_DECLARE_NV_MX_FORWARD(name) \
std::tuple<torch::Tensor, torch::Tensor> name( \
    torch::Tensor q4, torch::Tensor q4_scale, \
    torch::Tensor k4, torch::Tensor k4_scale, \
    torch::Tensor v4, torch::Tensor v4_scale, \
    torch::Tensor q8, torch::Tensor q8_scale, \
    torch::Tensor k8, torch::Tensor k8_scale, \
    torch::Tensor v8, torch::Tensor v8_scale, \
    torch::Tensor q16, torch::Tensor k16, torch::Tensor v16, \
    torch::Tensor block_ids, torch::Tensor nvfp4_counts, \
    torch::Tensor middle_counts, torch::Tensor fp16_counts, \
    torch::Tensor valid_k_counts, torch::Tensor q_global_scale, \
    torch::Tensor k_global_scale, torch::Tensor v_global_scale, \
    int64_t fp16_prefix_blocks, double softmax_scale, \
    bool active_fp16 = true)

MPA_DECLARE_NV_MX_FORWARD(sm120_q64_nv_mx_fp16_attention_forward);
MPA_DECLARE_NV_MX_FORWARD(
    sm120_q128_nv_mx_fp16_attention_forward);
#undef MPA_DECLARE_NV_MX_FORWARD

#define MPA_DECLARE_NV_INT8_FORWARD(name) \
std::tuple<torch::Tensor, torch::Tensor> name( \
    torch::Tensor q4, torch::Tensor q4_scale, \
    torch::Tensor k4, torch::Tensor k4_scale, \
    torch::Tensor v4, torch::Tensor v4_scale, \
    torch::Tensor q8, torch::Tensor q8_scale, \
    torch::Tensor k8, torch::Tensor k8_scale, \
    torch::Tensor v8, torch::Tensor v8_scale, \
    torch::Tensor q16, torch::Tensor k16, torch::Tensor v16, \
    torch::Tensor block_ids, torch::Tensor nvfp4_counts, \
    torch::Tensor middle_counts, torch::Tensor fp16_counts, \
    torch::Tensor valid_k_counts, torch::Tensor q_global_scale, \
    torch::Tensor k_global_scale, torch::Tensor v_global_scale, \
    int64_t fp16_prefix_blocks, double softmax_scale, \
    bool active_fp16 = true)

MPA_DECLARE_NV_INT8_FORWARD(sm120_q64_nv_int8_fp16_attention_forward);
MPA_DECLARE_NV_INT8_FORWARD(sm120_q128_nv_int8_fp16_attention_forward);
#undef MPA_DECLARE_NV_INT8_FORWARD
