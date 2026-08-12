/*
 * Native Q64 x K64 FP16 attention host dispatch for controlled Sol-H3
 * alignment experiments on SM89.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <tuple>

#include "api.h"
#include "k64_attention_decl.cuh"

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(
      tensor.device() == reference.device(), name,
      " must be on the same CUDA device as query");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> k64_fp16_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_ids,
    torch::Tensor block_counts,
    torch::Tensor valid_k_counts,
    double softmax_scale) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&query, "query"),
           std::pair<const torch::Tensor*, const char*>(&key, "key"),
           std::pair<const torch::Tensor*, const char*>(&value, "value"),
           std::pair<const torch::Tensor*, const char*>(&block_ids, "block_ids"),
           std::pair<const torch::Tensor*, const char*>(&block_counts, "block_counts"),
           std::pair<const torch::Tensor*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, query, item.second);
  }
  TORCH_CHECK(query.dim() == 4, "query must have shape [B,Hq,Q,D]");
  TORCH_CHECK(key.dim() == 4 && value.dim() == 4,
              "key/value must have shape [B,Hkv,K,D]");
  TORCH_CHECK(query.scalar_type() == at::ScalarType::Half,
              "query must be FP16");
  TORCH_CHECK(key.scalar_type() == at::ScalarType::Half,
              "key must be FP16");
  TORCH_CHECK(value.scalar_type() == at::ScalarType::Half,
              "value must be FP16");
  TORCH_CHECK(key.sizes() == value.sizes(), "key/value shapes must match");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0,
              "query dimensions must be positive");
  TORCH_CHECK(key.size(0) == query.size(0), "query/key batch mismatch");
  TORCH_CHECK(key.size(2) > 0 && key.size(2) % 64 == 0,
              "K must be a positive multiple of 64 physical slots");
  TORCH_CHECK(query.size(3) == 128 && key.size(3) == 128,
              "the initial native K64 specialization requires head_dim=128");
  TORCH_CHECK(query.size(1) % key.size(1) == 0,
              "query heads must be divisible by KV heads");
  TORCH_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  TORCH_CHECK(block_ids.scalar_type() == at::ScalarType::Int,
              "block_ids must be int32");
  TORCH_CHECK(block_counts.scalar_type() == at::ScalarType::Int,
              "block_counts must be int32");
  TORCH_CHECK(valid_k_counts.scalar_type() == at::ScalarType::Int,
              "valid_k_counts must be int32");
  const int64_t query_blocks = (query.size(2) + 63) / 64;
  const int64_t key_blocks = key.size(2) / 64;
  TORCH_CHECK(
      block_ids.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(1), query_blocks, key_blocks}),
      "block_ids must have shape [B,Hq,ceil(Q/64),K/64]");
  TORCH_CHECK(
      block_counts.sizes() ==
          torch::IntArrayRef({query.size(0), query.size(1), query_blocks}),
      "block_counts must have shape [B,Hq,ceil(Q/64)]");
  TORCH_CHECK(
      valid_k_counts.sizes() ==
          torch::IntArrayRef({query.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");

  auto output = torch::empty_like(query);
  auto lse = torch::empty(
      {query.size(0), query.size(1), query.size(2)},
      query.options().dtype(at::ScalarType::Float));

  c10::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  launch_mixed_attention_sm89_k64<128, false, true, false>(
      nullptr, nullptr, nullptr,
      reinterpret_cast<half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<half*>(key.data_ptr<at::Half>()),
      reinterpret_cast<half*>(value.data_ptr<at::Half>()), nullptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      nullptr, nullptr, block_ids.data_ptr<int32_t>(),
      block_counts.data_ptr<int32_t>(), nullptr, nullptr, nullptr,
      valid_k_counts.data_ptr<int32_t>(), lse.data_ptr<float>(),
      0,
      static_cast<uint32_t>(query.size(0)),
      static_cast<uint32_t>(query.size(2)),
      static_cast<uint32_t>(key.size(2)), 0,
      static_cast<uint32_t>(query.size(1)),
      static_cast<uint32_t>(key.size(1)),
      static_cast<float>(softmax_scale), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

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
    double softmax_scale) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q8, "q8"),
           std::pair<const torch::Tensor*, const char*>(&k8, "k8"),
           std::pair<const torch::Tensor*, const char*>(&v8, "v8"),
           std::pair<const torch::Tensor*, const char*>(&q16, "q16"),
           std::pair<const torch::Tensor*, const char*>(&k16, "k16"),
           std::pair<const torch::Tensor*, const char*>(&v16, "v16"),
           std::pair<const torch::Tensor*, const char*>(
               &fp8_block_ids, "fp8_block_ids"),
           std::pair<const torch::Tensor*, const char*>(
               &fp8_block_counts, "fp8_block_counts"),
           std::pair<const torch::Tensor*, const char*>(
               &fp16_block_ids, "fp16_block_ids"),
           std::pair<const torch::Tensor*, const char*>(
               &fp16_block_counts, "fp16_block_counts"),
           std::pair<const torch::Tensor*, const char*>(&q_scale, "q_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_scale, "k_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_scale, "v_scale"),
           std::pair<const torch::Tensor*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q16, item.second);
  }
  TORCH_CHECK(q16.dim() == 4, "q16 must have shape [B,Hq,Q,D]");
  TORCH_CHECK(k16.dim() == 4 && v16.dim() == 4,
              "k16/v16 must have shape [B,Hkv,K,D]");
  TORCH_CHECK(q16.scalar_type() == at::ScalarType::Half,
              "q16 must be FP16");
  TORCH_CHECK(k16.scalar_type() == at::ScalarType::Half,
              "k16 must be FP16");
  TORCH_CHECK(v16.scalar_type() == at::ScalarType::Half,
              "v16 must be FP16");
  TORCH_CHECK(k16.sizes() == v16.sizes(), "k16/v16 shapes must match");
  TORCH_CHECK(q16.size(0) > 0 && q16.size(1) > 0 && q16.size(2) > 0,
              "q16 dimensions must be positive");
  TORCH_CHECK(k16.size(0) == q16.size(0), "query/key batch mismatch");
  TORCH_CHECK(k16.size(2) > 0 && k16.size(2) % 64 == 0,
              "K must be a positive multiple of 64 physical slots");
  TORCH_CHECK(q16.size(3) == 128 && k16.size(3) == 128,
              "native mixed K64 currently requires head_dim=128");
  TORCH_CHECK(q16.size(1) % k16.size(1) == 0,
              "query heads must be divisible by KV heads");
  TORCH_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  TORCH_CHECK(q8.scalar_type() == at::ScalarType::Char,
              "q8 must be int8");
  TORCH_CHECK(k8.scalar_type() == at::ScalarType::Char,
              "k8 must be int8");
  TORCH_CHECK(v8.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "v8 must be float8_e4m3fn");
  TORCH_CHECK(q8.sizes() == q16.sizes(), "q8 must match q16 shape");
  TORCH_CHECK(k8.sizes() == k16.sizes(), "k8 must match k16 shape");
  TORCH_CHECK(
      v8.dim() == 4 && v8.size(0) == q16.size(0) &&
          v8.size(1) == k16.size(1) && v8.size(2) == q16.size(3),
      "v8 must have shape [B,Hkv,D,padded_K]");
  const int64_t padded_kv_len = v8.size(3);
  TORCH_CHECK(
      padded_kv_len >= ((k16.size(2) + 127) / 128) * 128 &&
          padded_kv_len % 128 == 0,
      "v8 token dimension must cover K and be padded to 128");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q_scale, "q_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_scale, "k_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_scale, "v_scale")}) {
    TORCH_CHECK(item.first->scalar_type() == at::ScalarType::Float,
                item.second, " must be FP32");
  }

  const int64_t query_blocks = (q16.size(2) + 63) / 64;
  const int64_t key_blocks = k16.size(2) / 64;
  TORCH_CHECK(
      q_scale.sizes() == torch::IntArrayRef(
          {q16.size(0), q16.size(1), query_blocks}),
      "q_scale must have Q64 shape [B,Hq,ceil(Q/64)]");
  TORCH_CHECK(
      k_scale.sizes() == torch::IntArrayRef(
          {q16.size(0), k16.size(1), key_blocks}),
      "k_scale must have K64 shape [B,Hkv,K/64]");
  TORCH_CHECK(
      v_scale.sizes() == torch::IntArrayRef(
          {q16.size(0), k16.size(1), q16.size(3)}),
      "v_scale must have shape [B,Hkv,D]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(
               &fp8_block_ids, "fp8_block_ids"),
           std::pair<const torch::Tensor*, const char*>(
               &fp16_block_ids, "fp16_block_ids")}) {
    TORCH_CHECK(item.first->scalar_type() == at::ScalarType::Int,
                item.second, " must be int32");
    TORCH_CHECK(
        item.first->sizes() == torch::IntArrayRef(
            {q16.size(0), q16.size(1), query_blocks, key_blocks}),
        item.second, " must have shape [B,Hq,ceil(Q/64),K/64]");
  }
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(
               &fp8_block_counts, "fp8_block_counts"),
           std::pair<const torch::Tensor*, const char*>(
               &fp16_block_counts, "fp16_block_counts")}) {
    TORCH_CHECK(item.first->scalar_type() == at::ScalarType::Int,
                item.second, " must be int32");
    TORCH_CHECK(
        item.first->sizes() == torch::IntArrayRef(
            {q16.size(0), q16.size(1), query_blocks}),
        item.second, " must have shape [B,Hq,ceil(Q/64)]");
  }
  TORCH_CHECK(valid_k_counts.scalar_type() == at::ScalarType::Int,
              "valid_k_counts must be int32");
  TORCH_CHECK(
      valid_k_counts.sizes() ==
          torch::IntArrayRef({q16.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");
  TORCH_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks,
      "fp16_prefix_blocks must be in [0,K/64]");
  TORCH_CHECK(
      fp8_block_ids.data_ptr<int32_t>() == fp16_block_ids.data_ptr<int32_t>(),
      "native mixed K64 requires one aliased compact FP8/FP16 route list");

  auto output = torch::empty_like(q16);
  auto lse = torch::empty(
      {q16.size(0), q16.size(1), q16.size(2)},
      q16.options().dtype(at::ScalarType::Float));
  c10::cuda::CUDAGuard device_guard(q16.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q16.get_device());
  launch_mixed_attention_sm89_k64<128, true, true, false>(
      q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
      reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
      reinterpret_cast<half*>(q16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(k16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(v16.data_ptr<at::Half>()), nullptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      fp8_block_ids.data_ptr<int32_t>(),
      fp8_block_counts.data_ptr<int32_t>(),
      fp8_block_ids.data_ptr<int32_t>(),
      fp16_block_counts.data_ptr<int32_t>(), q_scale.data_ptr<float>(),
      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
      valid_k_counts.data_ptr<int32_t>(), lse.data_ptr<float>(),
      static_cast<uint32_t>(fp16_prefix_blocks),
      static_cast<uint32_t>(q16.size(0)),
      static_cast<uint32_t>(q16.size(2)),
      static_cast<uint32_t>(k16.size(2)),
      static_cast<uint32_t>(padded_kv_len),
      static_cast<uint32_t>(q16.size(1)),
      static_cast<uint32_t>(k16.size(1)),
      static_cast<float>(softmax_scale), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}
