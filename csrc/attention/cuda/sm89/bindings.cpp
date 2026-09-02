/* SM89 MiniMax-H3 runtime operator registration. */

#include <pybind11/pybind11.h>
#include <torch/library.h>

#include "api.h"

TORCH_LIBRARY_FRAGMENT(mixed_attention, m) {
  m.def(
      "sm89_h3_draft_probability(Tensor q_pool, Tensor k_pool, "
      "Tensor? q_max_pool=None, Tensor? k_max_pool=None, "
      "float maxpool_weight=0.0) -> Tensor");
  m.def(
      "sm89_h3_route_precision(Tensor probability, int n16, int n8, int n4, "
      "Tensor? anchors, Tensor? anchor_ids, int anchor_count) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "sm89_h3_materialize_route("
      "Tensor logical_ids, Tensor low_counts, Tensor middle_counts, "
      "Tensor high_counts, int query_block_size, int prefix_blocks, "
      "int prefix_phase, bool prefix_first, bool has_high) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def("preprocess_v_fp8(Tensor value) -> (Tensor, Tensor)");
  m.def(
      "k64_fp16_attention_forward("
      "Tensor query, Tensor key, Tensor value, Tensor block_ids, "
      "Tensor block_counts, Tensor valid_k_counts, float softmax_scale) "
      "-> (Tensor, Tensor)");
  m.def(
      "q128_k64_fp16_attention_forward("
      "Tensor query, Tensor key, Tensor value, Tensor block_ids, "
      "Tensor block_counts, Tensor valid_k_counts, float softmax_scale) "
      "-> (Tensor, Tensor)");
  m.def(
      "k64_mixed_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q16, Tensor k16, Tensor v16, "
      "Tensor fp8_block_ids, Tensor fp8_block_counts, "
      "Tensor fp16_block_ids, Tensor fp16_block_counts, Tensor q_scale, "
      "Tensor k_scale, Tensor v_scale, Tensor valid_k_counts, "
      "int fp16_prefix_blocks, float softmax_scale) -> (Tensor, Tensor)");
  m.def(
      "q128_k64_mixed_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q16, Tensor k16, Tensor v16, "
      "Tensor fp8_block_ids, Tensor fp8_block_counts, "
      "Tensor fp16_block_ids, Tensor fp16_block_counts, Tensor q_scale, "
      "Tensor k_scale, Tensor v_scale, Tensor valid_k_counts, "
      "int fp16_prefix_blocks, float softmax_scale) -> (Tensor, Tensor)");
  m.def(
      "k64_smooth_mixed_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q16, Tensor k16, Tensor v16, "
      "Tensor key_mean, Tensor fp8_block_ids, Tensor fp8_block_counts, "
      "Tensor fp16_block_ids, Tensor fp16_block_counts, Tensor q_scale, "
      "Tensor k_scale, Tensor v_scale, Tensor valid_k_counts, "
      "int fp16_prefix_blocks, float softmax_scale) -> (Tensor, Tensor)");
  m.def(
      "q128_k64_smooth_mixed_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q16, Tensor k16, Tensor v16, "
      "Tensor key_mean, Tensor fp8_block_ids, Tensor fp8_block_counts, "
      "Tensor fp16_block_ids, Tensor fp16_block_counts, Tensor q_scale, "
      "Tensor k_scale, Tensor v_scale, Tensor valid_k_counts, "
      "int fp16_prefix_blocks, float softmax_scale) -> (Tensor, Tensor)");
  m.def(
      "k64_fp8_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor block_ids, "
      "Tensor block_counts, Tensor q_scale, Tensor k_scale, Tensor v_scale, "
      "Tensor valid_k_counts, float softmax_scale) -> (Tensor, Tensor)");
  m.def(
      "q128_k64_fp8_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor block_ids, "
      "Tensor block_counts, Tensor q_scale, Tensor k_scale, Tensor v_scale, "
      "Tensor valid_k_counts, float softmax_scale) -> (Tensor, Tensor)");
  m.def(
      "k64_smooth_fp8_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q16, Tensor key_mean, "
      "Tensor block_ids, Tensor block_counts, Tensor q_scale, Tensor k_scale, "
      "Tensor v_scale, Tensor valid_k_counts, float softmax_scale) "
      "-> (Tensor, Tensor)");
  m.def(
      "q128_k64_smooth_fp8_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q16, Tensor key_mean, "
      "Tensor block_ids, Tensor block_counts, Tensor q_scale, Tensor k_scale, "
      "Tensor v_scale, Tensor valid_k_counts, float softmax_scale) "
      "-> (Tensor, Tensor)");
  m.def(
      "prepare_sm89_prefix_q_int8(Tensor query, int prefix_tokens) "
      "-> (Tensor, Tensor)");
  m.def(
      "quantize_sm89_qk_int8(Tensor query, Tensor key, "
      "Tensor? key_mean=None, int query_block_size=64) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "sm89_q64_prefix_int8_attention_forward("
      "Tensor q8, Tensor k8, Tensor v8, Tensor q_scale, Tensor k_scale, "
      "Tensor v_scale, Tensor valid_k_counts, int prefix_tokens, "
      "float softmax_scale) -> Tensor");
  m.def(
      "pack_h3_k64_qkv_fp16(Tensor query, Tensor key, Tensor value, "
      "Tensor video_token_indices, Tensor video_slot_valid, int prefix_tokens) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "prepare_h3_sm89_int8_operands("
      "Tensor query, Tensor key, Tensor value, Tensor video_token_indices, "
      "Tensor video_slot_valid, Tensor video_valid_counts, int prefix_tokens, "
      "int query_block_size, bool smooth_k, bool has_maxpool=False) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, "
      "Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "assemble_h3_k64_output(Tensor prefix_output_bhsd, "
      "Tensor video_output_bhsd_fp16, Tensor video_inverse_indices, "
      "ScalarType? output_dtype=None, Tensor? low_counts=None, "
      "Tensor? middle_counts=None, Tensor? high_counts=None, "
      "int query_block_size=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(mixed_attention, CUDA, m) {
  m.impl("sm89_h3_draft_probability", &sm89_h3_draft_probability);
  m.impl("sm89_h3_route_precision", &sm89_h3_route_precision);
  m.impl("sm89_h3_materialize_route", &sm89_h3_materialize_route);
  m.impl("preprocess_v_fp8", &preprocess_v_fp8);
  m.impl("k64_fp16_attention_forward", &k64_fp16_attention_forward);
  m.impl(
      "q128_k64_fp16_attention_forward",
      &q128_k64_fp16_attention_forward);
  m.impl("k64_mixed_attention_forward", &k64_mixed_attention_forward);
  m.impl(
      "q128_k64_mixed_attention_forward",
      &q128_k64_mixed_attention_forward);
  m.impl(
      "k64_smooth_mixed_attention_forward",
      &k64_smooth_mixed_attention_forward);
  m.impl(
      "q128_k64_smooth_mixed_attention_forward",
      &q128_k64_smooth_mixed_attention_forward);
  m.impl("k64_fp8_attention_forward", &k64_fp8_attention_forward);
  m.impl(
      "q128_k64_fp8_attention_forward",
      &q128_k64_fp8_attention_forward);
  m.impl(
      "k64_smooth_fp8_attention_forward",
      &k64_smooth_fp8_attention_forward);
  m.impl(
      "q128_k64_smooth_fp8_attention_forward",
      &q128_k64_smooth_fp8_attention_forward);
  m.impl("prepare_sm89_prefix_q_int8", &prepare_sm89_prefix_q_int8);
  m.impl("quantize_sm89_qk_int8", &quantize_sm89_qk_int8);
  m.impl(
      "sm89_q64_prefix_int8_attention_forward",
      &sm89_q64_prefix_int8_attention_forward);
  m.impl("pack_h3_k64_qkv_fp16", &pack_h3_k64_qkv_fp16);
  m.impl(
      "prepare_h3_sm89_int8_operands",
      &prepare_h3_sm89_int8_operands);
  m.impl("assemble_h3_k64_output", &assemble_h3_k64_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  (void)module;
}
