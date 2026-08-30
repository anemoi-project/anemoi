/* SM120 Q64 all-FP16 operator registration. */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/library.h>

#include "api.h"

TORCH_LIBRARY_FRAGMENT(mixed_attention, m) {
  m.def(
      "sm120_h3_draft_probability(Tensor q_pool, Tensor k_pool, "
      "Tensor? q_max_pool=None, Tensor? k_max_pool=None, "
      "float maxpool_weight=0.0) -> Tensor");
  m.def(
      "sm120_h3_k_tail_r1_probability(Tensor q_pool, Tensor k_pool, "
      "Tensor packed_k, Tensor valid_counts, int prefix_blocks) -> Tensor");
  m.def(
      "sm120_h3_k_tail_r2_probability(Tensor q_pool, Tensor k_pool, "
      "Tensor packed_k, Tensor valid_counts, int prefix_blocks) -> Tensor");
  m.def(
      "sm120_h3_route_precision(Tensor probability, int n16, int n8, int n4, "
      "Tensor? anchors, Tensor? anchor_ids, int anchor_count) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "sm120_h3_materialize_route("
      "Tensor logical_ids, Tensor low_counts, Tensor middle_counts, "
      "Tensor high_counts, int query_block_size, int prefix_blocks, "
      "int prefix_phase, bool prefix_first, bool has_high) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "sm120_q64_fp16_attention_forward("
      "Tensor query, Tensor key, Tensor value, Tensor block_ids, "
      "Tensor block_counts, Tensor valid_k_counts, float softmax_scale) "
      "-> (Tensor, Tensor)");
  m.def(
      "sm120_q128_fp16_attention_forward("
      "Tensor query, Tensor key, Tensor value, Tensor block_ids, "
      "Tensor block_counts, Tensor valid_k_counts, float softmax_scale) "
      "-> (Tensor, Tensor)");
  m.def(
      "sm120_q64_int8_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q16, Tensor k16, Tensor v16, "
      "Tensor block_ids, Tensor int8_block_counts, Tensor fp16_block_counts, "
      "Tensor q_scale, Tensor k_scale, Tensor v_scale, "
      "Tensor valid_k_counts, int fp16_prefix_blocks, float softmax_scale, "
      "bool active_fp16=True) -> (Tensor, Tensor)");
  m.def(
      "sm120_q128_int8_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q16, Tensor k16, Tensor v16, "
      "Tensor block_ids, Tensor int8_block_counts, Tensor fp16_block_counts, "
      "Tensor q_scale, Tensor k_scale, Tensor v_scale, "
      "Tensor valid_k_counts, int fp16_prefix_blocks, float softmax_scale, "
      "bool active_fp16=True) -> (Tensor, Tensor)");
  m.def(
      "sm120_q64_prefix_int8_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q_scale, Tensor k_scale, "
      "Tensor v_scale, Tensor valid_k_counts, int prefix_tokens, "
      "float softmax_scale) -> Tensor");
  m.def(
      "sm120_q128_prefix_int8_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q_scale, Tensor k_scale, "
      "Tensor v_scale, Tensor valid_k_counts, int prefix_tokens, "
      "float softmax_scale) -> Tensor");
  m.def(
      "sm120_q64_mxfp8_attention_forward("
      "Tensor q_mxfp8, Tensor q_mxfp8_scale, Tensor k_mxfp8, "
      "Tensor k_mxfp8_scale, Tensor v_mxfp8, Tensor v_mxfp8_scale, "
      "Tensor q_fp16, Tensor k_fp16, Tensor v_fp16, Tensor block_ids, "
      "Tensor mxfp8_block_counts, Tensor fp16_block_counts, "
      "Tensor valid_k_counts, int fp16_prefix_blocks, "
      "float softmax_scale, bool active_fp16=True) -> (Tensor, Tensor)");
  m.def(
      "sm120_q128_mxfp8_attention_forward("
      "Tensor q_mxfp8, Tensor q_mxfp8_scale, Tensor k_mxfp8, "
      "Tensor k_mxfp8_scale, Tensor v_mxfp8, Tensor v_mxfp8_scale, "
      "Tensor q_fp16, Tensor k_fp16, Tensor v_fp16, Tensor block_ids, "
      "Tensor mxfp8_block_counts, Tensor fp16_block_counts, "
      "Tensor valid_k_counts, int fp16_prefix_blocks, "
      "float softmax_scale, bool active_fp16=True) -> (Tensor, Tensor)");
  m.def(
      "sm120_q128_mxfp8_compact_attention_forward("
      "Tensor q_mxfp8, Tensor q_mxfp8_scale, Tensor k_mxfp8, "
      "Tensor k_mxfp8_scale, Tensor v_mxfp8, Tensor v_mxfp8_scale, "
      "Tensor q_fp16, Tensor k_fp16, Tensor v_fp16, Tensor block_ids, "
      "Tensor mxfp8_block_counts, Tensor fp16_block_counts, "
      "Tensor valid_k_counts, int fp16_prefix_blocks, "
      "float softmax_scale, bool active_fp16=False) -> (Tensor, Tensor)");
  m.def(
      "prepare_h3_sm120_operands("
      "Tensor query, Tensor key, Tensor value, Tensor video_token_indices, "
      "Tensor video_slot_valid, Tensor video_valid_counts, int prefix_tokens, "
      "int query_block_size, bool has_nvfp4, bool has_int8, bool has_mxfp8, "
      "bool has_fp16, bool has_prefix_query_int8, bool has_maxpool, "
      "Tensor? q_global_scale, Tensor? k_global_scale, Tensor? v_global_scale) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, "
      "Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, "
      "Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, "
      "Tensor, Tensor)");
  m.def(
      "prepare_q64_nvfp4(Tensor query, Tensor key, Tensor value, "
      "Tensor q_global_scale, Tensor k_global_scale, "
      "Tensor v_global_scale) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "prepare_mxfp8(Tensor query, Tensor key, Tensor value) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "prepare_q128_nvfp4(Tensor query, Tensor key, Tensor value, "
      "Tensor q_global_scale, Tensor k_global_scale, "
      "Tensor v_global_scale) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "sm120_q128_nvfp4_attention_forward("
      "Tensor q_nvfp4, Tensor q_nvfp4_scale, Tensor k_nvfp4, "
      "Tensor k_nvfp4_scale, Tensor v_nvfp4, Tensor v_nvfp4_scale, "
      "Tensor q_fp16, Tensor k_fp16, Tensor v_fp16, Tensor block_ids, "
      "Tensor nvfp4_block_counts, Tensor fp16_block_counts, "
      "Tensor valid_k_counts, Tensor q_global_scale, "
      "Tensor k_global_scale, Tensor v_global_scale, "
      "int fp16_prefix_blocks, float softmax_scale, "
      "bool active_fp16=True) -> (Tensor, Tensor)");
  m.def(
      "sm120_q64_nvfp4_attention_forward("
      "Tensor q_nvfp4, Tensor q_nvfp4_scale, Tensor k_nvfp4, "
      "Tensor k_nvfp4_scale, Tensor v_nvfp4, Tensor v_nvfp4_scale, "
      "Tensor q_fp16, Tensor k_fp16, Tensor v_fp16, Tensor block_ids, "
      "Tensor nvfp4_block_counts, Tensor fp16_block_counts, "
      "Tensor valid_k_counts, Tensor q_global_scale, "
      "Tensor k_global_scale, Tensor v_global_scale, "
      "int fp16_prefix_blocks, float softmax_scale, "
      "bool active_fp16=True) -> (Tensor, Tensor)");
  m.def(
      "sm120_q64_nv_mx_fp16_attention_forward("
      "Tensor q4, Tensor q4_scale, Tensor k4, Tensor k4_scale, "
      "Tensor v4, Tensor v4_scale, Tensor q8, Tensor q8_scale, "
      "Tensor k8, Tensor k8_scale, Tensor v8, Tensor v8_scale, "
      "Tensor q16, Tensor k16, Tensor v16, Tensor block_ids, "
      "Tensor nvfp4_counts, Tensor middle_counts, Tensor fp16_counts, "
      "Tensor valid_k_counts, Tensor q_global_scale, "
      "Tensor k_global_scale, Tensor v_global_scale, "
      "int fp16_prefix_blocks, float softmax_scale, "
      "bool active_fp16=True) -> (Tensor, Tensor)");
  m.def(
      "sm120_q128_nv_mx_fp16_attention_forward("
      "Tensor q4, Tensor q4_scale, Tensor k4, Tensor k4_scale, "
      "Tensor v4, Tensor v4_scale, Tensor q8, Tensor q8_scale, "
      "Tensor k8, Tensor k8_scale, Tensor v8, Tensor v8_scale, "
      "Tensor q16, Tensor k16, Tensor v16, Tensor block_ids, "
      "Tensor nvfp4_counts, Tensor middle_counts, Tensor fp16_counts, "
      "Tensor valid_k_counts, Tensor q_global_scale, "
      "Tensor k_global_scale, Tensor v_global_scale, "
      "int fp16_prefix_blocks, float softmax_scale, "
      "bool active_fp16=True) -> (Tensor, Tensor)");
  m.def(
      "sm120_q64_nv_int8_fp16_attention_forward("
      "Tensor q4, Tensor q4_scale, Tensor k4, Tensor k4_scale, "
      "Tensor v4, Tensor v4_scale, Tensor q8, Tensor q8_scale, "
      "Tensor k8, Tensor k8_scale, Tensor v8, Tensor v8_scale, "
      "Tensor q16, Tensor k16, Tensor v16, Tensor block_ids, "
      "Tensor nvfp4_counts, Tensor middle_counts, Tensor fp16_counts, "
      "Tensor valid_k_counts, Tensor q_global_scale, "
      "Tensor k_global_scale, Tensor v_global_scale, "
      "int fp16_prefix_blocks, float softmax_scale, "
      "bool active_fp16=True) -> (Tensor, Tensor)");
  m.def(
      "sm120_q128_nv_int8_fp16_attention_forward("
      "Tensor q4, Tensor q4_scale, Tensor k4, Tensor k4_scale, "
      "Tensor v4, Tensor v4_scale, Tensor q8, Tensor q8_scale, "
      "Tensor k8, Tensor k8_scale, Tensor v8, Tensor v8_scale, "
      "Tensor q16, Tensor k16, Tensor v16, Tensor block_ids, "
      "Tensor nvfp4_counts, Tensor middle_counts, Tensor fp16_counts, "
      "Tensor valid_k_counts, Tensor q_global_scale, "
      "Tensor k_global_scale, Tensor v_global_scale, "
      "int fp16_prefix_blocks, float softmax_scale, "
      "bool active_fp16=True) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mixed_attention, CUDA, m) {
  m.impl("sm120_h3_draft_probability", &sm120_h3_draft_probability);
  m.impl(
      "sm120_h3_k_tail_r1_probability",
      &sm120_h3_k_tail_r1_probability);
  m.impl(
      "sm120_h3_k_tail_r2_probability",
      &sm120_h3_k_tail_r2_probability);
  m.impl("sm120_h3_route_precision", &sm120_h3_route_precision);
  m.impl("sm120_h3_materialize_route", &sm120_h3_materialize_route);
  m.impl(
      "sm120_q64_fp16_attention_forward",
      &sm120_q64_fp16_attention_forward);
  m.impl(
      "sm120_q128_fp16_attention_forward",
      &sm120_q128_fp16_attention_forward);
  m.impl(
      "sm120_q64_int8_attention_forward",
      &sm120_q64_int8_attention_forward);
  m.impl(
      "sm120_q128_int8_attention_forward",
      &sm120_q128_int8_attention_forward);
  m.impl(
      "sm120_q64_prefix_int8_attention_forward",
      &sm120_q64_prefix_int8_attention_forward);
  m.impl(
      "sm120_q128_prefix_int8_attention_forward",
      &sm120_q128_prefix_int8_attention_forward);
  m.impl(
      "sm120_q64_mxfp8_attention_forward",
      &sm120_q64_mxfp8_attention_forward);
  m.impl(
      "sm120_q128_mxfp8_attention_forward",
      &sm120_q128_mxfp8_attention_forward);
  m.impl(
      "sm120_q128_mxfp8_compact_attention_forward",
      &sm120_q128_mxfp8_compact_attention_forward);
  m.impl("prepare_h3_sm120_operands", &prepare_h3_sm120_operands);
  m.impl("prepare_q64_nvfp4", &prepare_q64_nvfp4);
  m.impl("prepare_q128_nvfp4", &prepare_q128_nvfp4);
  m.impl("prepare_mxfp8", &prepare_mxfp8);
  m.impl(
      "sm120_q64_nvfp4_attention_forward",
      &sm120_q64_nvfp4_attention_forward);
  m.impl(
      "sm120_q128_nvfp4_attention_forward",
      &sm120_q128_nvfp4_attention_forward);
  m.impl(
      "sm120_q64_nv_mx_fp16_attention_forward",
      &sm120_q64_nv_mx_fp16_attention_forward);
  m.impl(
      "sm120_q128_nv_mx_fp16_attention_forward",
      &sm120_q128_nv_mx_fp16_attention_forward);
  m.impl(
      "sm120_q128_nv_int8_fp16_attention_forward",
      &sm120_q128_nv_int8_fp16_attention_forward);
  m.impl(
      "sm120_q64_nv_int8_fp16_attention_forward",
      &sm120_q64_nv_int8_fp16_attention_forward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "sm120_q64_fp16_kernel_metadata",
      &sm120_q64_fp16_kernel_metadata);
  module.def(
      "sm120_q64_mxfp8_kernel_metadata",
      &sm120_q64_mxfp8_kernel_metadata);
}
