/* SM89 MiniMax-H3 runtime operator registration. */

#include <pybind11/pybind11.h>
#include <torch/library.h>

#include "api.h"

TORCH_LIBRARY_FRAGMENT(mixed_attention, m) {
  m.def("preprocess_v_fp8(Tensor value) -> (Tensor, Tensor)");
  m.def(
      "k64_fp16_attention_forward("
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
      "pack_h3_k64_qkv_fp16(Tensor query, Tensor key, Tensor value, "
      "Tensor video_token_indices, Tensor video_slot_valid, int prefix_tokens) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "assemble_h3_k64_output(Tensor prefix_output_bhsd, "
      "Tensor video_output_bhsd_fp16, Tensor video_inverse_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(mixed_attention, CUDA, m) {
  m.impl("preprocess_v_fp8", &preprocess_v_fp8);
  m.impl("k64_fp16_attention_forward", &k64_fp16_attention_forward);
  m.impl("k64_mixed_attention_forward", &k64_mixed_attention_forward);
  m.impl(
      "q128_k64_mixed_attention_forward",
      &q128_k64_mixed_attention_forward);
  m.impl("k64_fp8_attention_forward", &k64_fp8_attention_forward);
  m.impl(
      "q128_k64_fp8_attention_forward",
      &q128_k64_fp8_attention_forward);
  m.impl("pack_h3_k64_qkv_fp16", &pack_h3_k64_qkv_fp16);
  m.impl("assemble_h3_k64_output", &assemble_h3_k64_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  (void)module;
}
