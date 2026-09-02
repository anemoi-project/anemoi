/*
 * Native Q64 x K64 FP16 attention host dispatch for controlled Sol-H3
 * alignment experiments on SM120.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "api.h"
#include "q64_attention_decl.cuh"

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

template <uint32_t QueryBlock>
std::tuple<torch::Tensor, torch::Tensor> fp16_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_ids,
    torch::Tensor block_counts,
    torch::Tensor valid_k_counts,
    double softmax_scale) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
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
  if constexpr (QueryBlock == 128) {
    TORCH_CHECK(query.size(2) % 128 == 0, "Q must be a multiple of 128");
  }
  TORCH_CHECK(key.size(0) == query.size(0), "query/key batch mismatch");
  TORCH_CHECK(key.size(2) > 0 && key.size(2) % 64 == 0,
              "K must be a positive multiple of 64 physical slots");
  TORCH_CHECK(query.size(3) == 128 && key.size(3) == 128,
              "native K64 attention requires head_dim=128");
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
  const int64_t query_blocks =
      (query.size(2) + QueryBlock - 1) / QueryBlock;
  const int64_t key_blocks = key.size(2) / 64;
  TORCH_CHECK(
      block_ids.sizes() == torch::IntArrayRef(
          {query.size(0), query.size(1), query_blocks, key_blocks}),
      "block_ids must have shape [B,Hq,ceil(Q/query_block),K/64]");
  TORCH_CHECK(
      block_counts.sizes() ==
          torch::IntArrayRef({query.size(0), query.size(1), query_blocks}),
      "block_counts must have shape [B,Hq,ceil(Q/query_block)]");
  TORCH_CHECK(
      valid_k_counts.sizes() ==
          torch::IntArrayRef({query.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");

  auto output = torch::empty_like(query);
  auto lse = torch::empty(
      {0},
      query.options().dtype(at::ScalarType::Float));

  c10::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  if constexpr (QueryBlock == 64) {
    launch_mixed_attention_sm120_q64<128, false, true, false>(
        nullptr, nullptr, nullptr,
        reinterpret_cast<half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value.data_ptr<at::Half>()), nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        nullptr, nullptr, block_ids.data_ptr<int32_t>(),
        block_counts.data_ptr<int32_t>(), nullptr, nullptr, nullptr,
        valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
        static_cast<uint32_t>(query.size(0)),
        static_cast<uint32_t>(query.size(2)),
        static_cast<uint32_t>(key.size(2)), 0,
        static_cast<uint32_t>(query.size(1)),
        static_cast<uint32_t>(key.size(1)),
        static_cast<float>(softmax_scale), stream);
  } else {
    launch_mixed_attention_sm120_q128_fp16<128, false, true, false>(
        nullptr, nullptr, nullptr,
        reinterpret_cast<half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value.data_ptr<at::Half>()), nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        nullptr, nullptr, block_ids.data_ptr<int32_t>(),
        block_counts.data_ptr<int32_t>(), nullptr, nullptr, nullptr,
        valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
        static_cast<uint32_t>(query.size(0)),
        static_cast<uint32_t>(query.size(2)),
        static_cast<uint32_t>(key.size(2)), 0,
        static_cast<uint32_t>(query.size(1)),
        static_cast<uint32_t>(key.size(1)),
        static_cast<float>(softmax_scale), stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> sm120_q64_fp16_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_ids,
    torch::Tensor block_counts,
    torch::Tensor valid_k_counts,
    double softmax_scale) {
  return fp16_attention_forward<64>(
      query, key, value, block_ids, block_counts, valid_k_counts,
      softmax_scale);
}

std::tuple<torch::Tensor, torch::Tensor> sm120_q128_fp16_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_ids,
    torch::Tensor block_counts,
    torch::Tensor valid_k_counts,
    double softmax_scale) {
  return fp16_attention_forward<128>(
      query, key, value, block_ids, block_counts, valid_k_counts,
      softmax_scale);
}

template <uint32_t QueryBlock>
std::tuple<torch::Tensor, torch::Tensor> int8_attention_forward(
    torch::Tensor q8,
    torch::Tensor k8,
    torch::Tensor v8,
    torch::Tensor q16,
    torch::Tensor k16,
    torch::Tensor v16,
    torch::Tensor block_ids,
    torch::Tensor int8_block_counts,
    torch::Tensor fp16_block_counts,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale,
    bool active_fp16) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q8, "q8"),
           std::pair<const torch::Tensor*, const char*>(&k8, "k8"),
           std::pair<const torch::Tensor*, const char*>(&v8, "v8"),
           std::pair<const torch::Tensor*, const char*>(&q16, "q16"),
           std::pair<const torch::Tensor*, const char*>(&k16, "k16"),
           std::pair<const torch::Tensor*, const char*>(&v16, "v16"),
           std::pair<const torch::Tensor*, const char*>(
               &block_ids, "block_ids"),
           std::pair<const torch::Tensor*, const char*>(
               &int8_block_counts, "int8_block_counts"),
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

  TORCH_CHECK(
      q16.dim() == 4 && k16.dim() == 4 && v16.dim() == 4,
      "q16/k16/v16 must have shape [B,H,Q_or_K,D]");
  TORCH_CHECK(
      q16.scalar_type() == at::ScalarType::Half &&
          k16.scalar_type() == at::ScalarType::Half &&
          v16.scalar_type() == at::ScalarType::Half,
      "q16/k16/v16 must be FP16");
  TORCH_CHECK(k16.sizes() == v16.sizes(), "k16/v16 shapes must match");
  TORCH_CHECK(
      q16.size(0) > 0 && q16.size(1) > 0 && q16.size(2) > 0 &&
          q16.size(2) % QueryBlock == 0,
      "Q must be positive and divisible by the query block");
  TORCH_CHECK(
      k16.size(0) == q16.size(0) && k16.size(2) > 0 &&
          k16.size(2) % 64 == 0,
      "K must share the batch and contain complete physical K64 slots");
  TORCH_CHECK(
      q16.size(3) == 128 && k16.size(3) == 128,
      "native INT8 K64 attention requires head_dim=128");
  TORCH_CHECK(
      q16.size(1) % k16.size(1) == 0,
      "query heads must be divisible by KV heads");
  TORCH_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  TORCH_CHECK(
      q8.scalar_type() == at::ScalarType::Char &&
          k8.scalar_type() == at::ScalarType::Char,
      "q8/k8 must be int8");
  TORCH_CHECK(
      v8.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "v8 must be float8_e4m3fn");
  TORCH_CHECK(q8.sizes() == q16.sizes(), "q8 must match q16 shape");
  TORCH_CHECK(k8.sizes() == k16.sizes(), "k8 must match k16 shape");
  TORCH_CHECK(
      v8.dim() == 4 && v8.size(0) == q16.size(0) &&
          v8.size(1) == k16.size(1) && v8.size(2) == 128,
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
    TORCH_CHECK(
        item.first->scalar_type() == at::ScalarType::Float,
        item.second, " must be FP32");
  }
  const int64_t query_blocks = q16.size(2) / QueryBlock;
  const int64_t key_blocks = k16.size(2) / 64;
  TORCH_CHECK(
      q_scale.sizes() == torch::IntArrayRef(
          {q16.size(0), q16.size(1), query_blocks}),
      "q_scale must have shape [B,Hq,Q/query_block]");
  TORCH_CHECK(
      k_scale.sizes() == torch::IntArrayRef(
          {q16.size(0), k16.size(1), key_blocks}),
      "k_scale must have shape [B,Hkv,K/64]");
  TORCH_CHECK(
      v_scale.sizes() == torch::IntArrayRef(
          {q16.size(0), k16.size(1), q16.size(3)}),
      "v_scale must have shape [B,Hkv,D]");

  TORCH_CHECK(
      block_ids.scalar_type() == at::ScalarType::Int &&
          block_ids.sizes() == torch::IntArrayRef(
              {q16.size(0), q16.size(1), query_blocks, key_blocks}),
      "block_ids must have shape [B,Hq,Q/query_block,K/64]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(
               &int8_block_counts, "int8_block_counts")}) {
    TORCH_CHECK(
        item.first->scalar_type() == at::ScalarType::Int &&
            item.first->sizes() == torch::IntArrayRef(
                {q16.size(0), q16.size(1), query_blocks}),
        item.second, " must have shape [B,Hq,Q/query_block]");
  }
  TORCH_CHECK(
      fp16_block_counts.scalar_type() == at::ScalarType::Int &&
          (active_fp16
               ? fp16_block_counts.sizes() == torch::IntArrayRef(
                     {q16.size(0), q16.size(1), query_blocks})
               : fp16_block_counts.numel() == 0),
      "active FP16 counts must match route rows; inactive FP16 requires "
      "an empty count tensor");
  TORCH_CHECK(
      valid_k_counts.scalar_type() == at::ScalarType::Int &&
          valid_k_counts.sizes() == torch::IntArrayRef(
              {q16.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");
  TORCH_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks,
      "fp16_prefix_blocks must be in [0,K/64]");
  TORCH_CHECK(
      active_fp16 || fp16_prefix_blocks == 0,
      "inactive FP16 requires zero prefix stages");

  auto output = torch::empty_like(q16);
  auto lse = torch::empty(
      {0},
      q16.options().dtype(at::ScalarType::Float));
  c10::cuda::CUDAGuard device_guard(q16.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q16.get_device());

#define MPA_LAUNCH_INT8(launcher) \
  launcher( \
      q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(), \
      reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()), \
      reinterpret_cast<half*>(q16.data_ptr<at::Half>()), \
      reinterpret_cast<half*>(k16.data_ptr<at::Half>()), \
      reinterpret_cast<half*>(v16.data_ptr<at::Half>()), nullptr, \
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), \
      block_ids.data_ptr<int32_t>(), int8_block_counts.data_ptr<int32_t>(), \
      block_ids.data_ptr<int32_t>(), fp16_block_counts.data_ptr<int32_t>(), \
      q_scale.data_ptr<float>(), k_scale.data_ptr<float>(), \
      v_scale.data_ptr<float>(), valid_k_counts.data_ptr<int32_t>(), \
      nullptr, static_cast<uint32_t>(fp16_prefix_blocks), \
      static_cast<uint32_t>(q16.size(0)), \
      static_cast<uint32_t>(q16.size(2)), \
      static_cast<uint32_t>(k16.size(2)), \
      static_cast<uint32_t>(padded_kv_len), \
      static_cast<uint32_t>(q16.size(1)), \
      static_cast<uint32_t>(k16.size(1)), \
      static_cast<float>(softmax_scale), stream)

  if constexpr (QueryBlock == 64) {
    auto launcher = active_fp16
        ? launch_mixed_attention_sm120_q64_int8_fp16<128, true, true, false>
        : launch_mixed_attention_sm120_q64_int8<128, true, false, false>;
    MPA_LAUNCH_INT8(launcher);
  } else {
    auto launcher = active_fp16
        ? launch_mixed_attention_sm120_q128_int8<128, true, true, false>
        : launch_mixed_attention_sm120_q128_int8<128, true, false, false>;
    MPA_LAUNCH_INT8(launcher);
  }
#undef MPA_LAUNCH_INT8

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

std::tuple<torch::Tensor, torch::Tensor> sm120_q64_int8_attention_forward(
    torch::Tensor q8, torch::Tensor k8, torch::Tensor v8,
    torch::Tensor q16, torch::Tensor k16, torch::Tensor v16,
    torch::Tensor block_ids, torch::Tensor int8_block_counts,
    torch::Tensor fp16_block_counts, torch::Tensor q_scale,
    torch::Tensor k_scale, torch::Tensor v_scale,
    torch::Tensor valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16) {
  return int8_attention_forward<64>(
      q8, k8, v8, q16, k16, v16, block_ids, int8_block_counts,
      fp16_block_counts, q_scale, k_scale, v_scale, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16);
}

std::tuple<torch::Tensor, torch::Tensor> sm120_q128_int8_attention_forward(
    torch::Tensor q8, torch::Tensor k8, torch::Tensor v8,
    torch::Tensor q16, torch::Tensor k16, torch::Tensor v16,
    torch::Tensor block_ids, torch::Tensor int8_block_counts,
    torch::Tensor fp16_block_counts, torch::Tensor q_scale,
    torch::Tensor k_scale, torch::Tensor v_scale,
    torch::Tensor valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16) {
  return int8_attention_forward<128>(
      q8, k8, v8, q16, k16, v16, block_ids, int8_block_counts,
      fp16_block_counts, q_scale, k_scale, v_scale, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16);
}

template <uint32_t QueryBlock>
torch::Tensor prefix_int8_attention_forward(
    torch::Tensor q8,
    torch::Tensor k8,
    torch::Tensor v8,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor valid_k_counts,
    int64_t prefix_tokens,
    double softmax_scale) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q8, "q8"),
           std::pair<const torch::Tensor*, const char*>(&k8, "k8"),
           std::pair<const torch::Tensor*, const char*>(&v8, "v8"),
           std::pair<const torch::Tensor*, const char*>(&q_scale, "q_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_scale, "k_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_scale, "v_scale"),
           std::pair<const torch::Tensor*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q8, item.second);
  }
  TORCH_CHECK(
      q8.scalar_type() == at::ScalarType::Char &&
          k8.scalar_type() == at::ScalarType::Char,
      "q8/k8 must be int8");
  TORCH_CHECK(
      v8.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "v8 must be float8_e4m3fn");
  TORCH_CHECK(
      q8.dim() == 4 && q8.size(0) > 0 && q8.size(1) > 0 &&
          q8.size(2) > 0 && q8.size(2) % QueryBlock == 0 &&
          q8.size(3) == 128,
      "q8 must have shape [B,Hq,Q,128] with Q divisible by query block");
  TORCH_CHECK(
      k8.dim() == 4 && k8.size(0) == q8.size(0) && k8.size(1) > 0 &&
          k8.size(2) > 0 && k8.size(2) % 64 == 0 && k8.size(3) == 128 &&
          q8.size(1) % k8.size(1) == 0,
      "k8 must have compatible [B,Hkv,K,128] K64 layout");
  TORCH_CHECK(
      prefix_tokens > 0 && prefix_tokens <= q8.size(2),
      "prefix_tokens must be in (0,Q]");
  TORCH_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  const int64_t query_blocks = q8.size(2) / QueryBlock;
  const int64_t key_blocks = k8.size(2) / 64;
  TORCH_CHECK(
      v8.dim() == 4 && v8.size(0) == q8.size(0) &&
          v8.size(1) == k8.size(1) && v8.size(2) == 128 &&
          v8.size(3) >= ((k8.size(2) + 127) / 128) * 128 &&
          v8.size(3) % 128 == 0,
      "v8 must have verified [B,Hkv,128,padded_K] layout");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q_scale, "q_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_scale, "k_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_scale, "v_scale")}) {
    TORCH_CHECK(
        item.first->scalar_type() == at::ScalarType::Float,
        item.second, " must be FP32");
  }
  TORCH_CHECK(
      q_scale.sizes() == torch::IntArrayRef(
          {q8.size(0), q8.size(1), query_blocks}),
      "q_scale must have shape [B,Hq,Q/query_block]");
  TORCH_CHECK(
      k_scale.sizes() == torch::IntArrayRef(
          {q8.size(0), k8.size(1), key_blocks}),
      "k_scale must have shape [B,Hkv,K/64]");
  TORCH_CHECK(
      v_scale.sizes() == torch::IntArrayRef(
          {q8.size(0), k8.size(1), 128}),
      "v_scale must have shape [B,Hkv,128]");
  TORCH_CHECK(
      valid_k_counts.scalar_type() == at::ScalarType::Int &&
          valid_k_counts.sizes() ==
              torch::IntArrayRef({q8.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");

  auto output = torch::empty(
      q8.sizes(), q8.options().dtype(at::ScalarType::Half));
  c10::cuda::CUDAGuard device_guard(q8.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q8.get_device());
  auto launcher = QueryBlock == 64
      ? launch_mixed_attention_sm120_q64_int8_dense<128, true, false, false>
      : launch_mixed_attention_sm120_q128_int8_dense<128, true, false, false>;
  launcher(
      q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
      reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
      nullptr, nullptr, nullptr, nullptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      nullptr, nullptr, nullptr, nullptr,
      q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
      v_scale.data_ptr<float>(), valid_k_counts.data_ptr<int32_t>(),
      nullptr, 0, static_cast<uint32_t>(q8.size(0)),
      static_cast<uint32_t>(q8.size(2)),
      static_cast<uint32_t>(k8.size(2)),
      static_cast<uint32_t>(v8.size(3)),
      static_cast<uint32_t>(q8.size(1)),
      static_cast<uint32_t>(k8.size(1)),
      static_cast<float>(softmax_scale), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output.narrow(2, 0, prefix_tokens);
}

torch::Tensor sm120_q64_prefix_int8_attention_forward(
    torch::Tensor q8, torch::Tensor k8, torch::Tensor v8,
    torch::Tensor q_scale, torch::Tensor k_scale, torch::Tensor v_scale,
    torch::Tensor valid_k_counts, int64_t prefix_tokens,
    double softmax_scale) {
  return prefix_int8_attention_forward<64>(
      q8, k8, v8, q_scale, k_scale, v_scale, valid_k_counts,
      prefix_tokens, softmax_scale);
}

torch::Tensor sm120_q128_prefix_int8_attention_forward(
    torch::Tensor q8, torch::Tensor k8, torch::Tensor v8,
    torch::Tensor q_scale, torch::Tensor k_scale, torch::Tensor v_scale,
    torch::Tensor valid_k_counts, int64_t prefix_tokens,
    double softmax_scale) {
  return prefix_int8_attention_forward<128>(
      q8, k8, v8, q_scale, k_scale, v_scale, valid_k_counts,
      prefix_tokens, softmax_scale);
}

template <uint32_t QueryBlock, bool CompactSequential = false>
std::tuple<torch::Tensor, torch::Tensor> mxfp8_attention_forward(
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
    bool active_fp16) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q_mxfp8, "q_mxfp8"),
           std::pair<const torch::Tensor*, const char*>(
               &q_mxfp8_scale, "q_mxfp8_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_mxfp8, "k_mxfp8"),
           std::pair<const torch::Tensor*, const char*>(
               &k_mxfp8_scale, "k_mxfp8_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_mxfp8, "v_mxfp8"),
           std::pair<const torch::Tensor*, const char*>(
               &v_mxfp8_scale, "v_mxfp8_scale"),
           std::pair<const torch::Tensor*, const char*>(&q_fp16, "q_fp16"),
           std::pair<const torch::Tensor*, const char*>(&k_fp16, "k_fp16"),
           std::pair<const torch::Tensor*, const char*>(&v_fp16, "v_fp16"),
           std::pair<const torch::Tensor*, const char*>(&block_ids, "block_ids"),
           std::pair<const torch::Tensor*, const char*>(
               &mxfp8_block_counts, "mxfp8_block_counts"),
           std::pair<const torch::Tensor*, const char*>(
               &fp16_block_counts, "fp16_block_counts"),
           std::pair<const torch::Tensor*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q_fp16, item.second);
  }
  TORCH_CHECK(
      q_fp16.dim() == 4 && k_fp16.dim() == 4 && v_fp16.dim() == 4,
      "FP16 operands must have shape [B,H,S,D]");
  TORCH_CHECK(
      q_fp16.scalar_type() == at::ScalarType::Half &&
          k_fp16.scalar_type() == at::ScalarType::Half &&
          v_fp16.scalar_type() == at::ScalarType::Half,
      "FP16 operands must be FP16");
  TORCH_CHECK(k_fp16.sizes() == v_fp16.sizes(), "FP16 K/V shapes must match");
  TORCH_CHECK(
      q_fp16.size(0) > 0 && q_fp16.size(1) > 0 &&
          q_fp16.size(2) > 0 && q_fp16.size(2) % QueryBlock == 0,
      "MXFP8 Q length must be a positive multiple of the query block");
  TORCH_CHECK(
      k_fp16.size(0) == q_fp16.size(0) && k_fp16.size(2) > 0 &&
          k_fp16.size(2) % 64 == 0,
      "MXFP8 K length must be a positive multiple of 64 with matching batch");
  TORCH_CHECK(
      q_fp16.size(3) == 128 && k_fp16.size(3) == 128 &&
          q_fp16.size(1) % k_fp16.size(1) == 0,
      "SM120 MXFP8 requires D128 and divisible Q/KV heads");
  TORCH_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q_mxfp8, "q_mxfp8"),
           std::pair<const torch::Tensor*, const char*>(
               &q_mxfp8_scale, "q_mxfp8_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_mxfp8, "k_mxfp8"),
           std::pair<const torch::Tensor*, const char*>(
               &k_mxfp8_scale, "k_mxfp8_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_mxfp8, "v_mxfp8"),
           std::pair<const torch::Tensor*, const char*>(
               &v_mxfp8_scale, "v_mxfp8_scale")}) {
    TORCH_CHECK(
        item.first->scalar_type() == at::ScalarType::Byte,
        item.second, " must be uint8");
  }
  TORCH_CHECK(q_mxfp8.sizes() == q_fp16.sizes(), "q_mxfp8 must match q_fp16");
  TORCH_CHECK(k_mxfp8.sizes() == k_fp16.sizes(), "k_mxfp8 must match k_fp16");
  TORCH_CHECK(
      q_mxfp8_scale.sizes() == torch::IntArrayRef(
          {q_fp16.size(0), q_fp16.size(1), q_fp16.size(2), 4}),
      "q_mxfp8_scale must have shape [B,Hq,Q,4]");
  TORCH_CHECK(
      k_mxfp8_scale.sizes() == torch::IntArrayRef(
          {k_fp16.size(0), k_fp16.size(1), k_fp16.size(2), 4}),
      "k_mxfp8_scale must have shape [B,Hkv,K,4]");
  TORCH_CHECK(
      v_mxfp8.dim() == 4 && v_mxfp8.size(0) == k_fp16.size(0) &&
          v_mxfp8.size(1) == k_fp16.size(1) &&
          v_mxfp8.size(2) == 128 &&
          v_mxfp8.size(3) >= k_fp16.size(2) &&
          v_mxfp8.size(3) % 64 == 0,
      "v_mxfp8 must have shape [B,Hkv,128,padded_K]");
  const int64_t query_blocks = q_fp16.size(2) / QueryBlock;
  const int64_t key_blocks = k_fp16.size(2) / 64;
  TORCH_CHECK(
      v_mxfp8_scale.sizes() == torch::IntArrayRef(
          {k_fp16.size(0), k_fp16.size(1), key_blocks, 256}),
      "v_mxfp8_scale must have K64 consumer shape [B,Hkv,K/64,256]");

  TORCH_CHECK(block_ids.scalar_type() == at::ScalarType::Int,
              "block_ids must be int32");
  TORCH_CHECK(mxfp8_block_counts.scalar_type() == at::ScalarType::Int,
              "mxfp8_block_counts must be int32");
  TORCH_CHECK(fp16_block_counts.scalar_type() == at::ScalarType::Int,
              "fp16_block_counts must be int32");
  TORCH_CHECK(valid_k_counts.scalar_type() == at::ScalarType::Int,
              "valid_k_counts must be int32");
  TORCH_CHECK(
      block_ids.sizes() == torch::IntArrayRef(
          {q_fp16.size(0), q_fp16.size(1), query_blocks, key_blocks}),
      "block_ids must have shape [B,Hq,Q/query_block,K/64]");
  TORCH_CHECK(
      mxfp8_block_counts.sizes() == torch::IntArrayRef(
          {q_fp16.size(0), q_fp16.size(1), query_blocks}),
              "mxfp8_block_counts must have shape [B,Hq,Q/query_block]");
  TORCH_CHECK(
      active_fp16
          ? fp16_block_counts.sizes() == torch::IntArrayRef(
                {q_fp16.size(0), q_fp16.size(1), query_blocks})
          : fp16_block_counts.numel() == 0,
      "active FP16 counts must match route rows; inactive FP16 requires "
      "an empty count tensor");
  TORCH_CHECK(
      valid_k_counts.sizes() == torch::IntArrayRef(
          {q_fp16.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");
  TORCH_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks,
      "fp16_prefix_blocks must be in [0,K/64]");
  TORCH_CHECK(
      active_fp16 || fp16_prefix_blocks == 0,
      "inactive FP16 requires zero prefix stages");
  auto output = torch::empty_like(q_fp16);
  auto lse = torch::empty(
      {0},
      q_fp16.options().dtype(at::ScalarType::Float));
  c10::cuda::CUDAGuard device_guard(q_fp16.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_fp16.get_device());

  if constexpr (CompactSequential) {
    static_assert(QueryBlock == 128);
    TORCH_CHECK(!active_fp16, "compact MXFP8 is a pure compute ceiling");
    launch_mixed_attention_sm120_q128_mxfp8_compact<
        128, true, false, false>(
        reinterpret_cast<int8_t*>(q_mxfp8.data_ptr<uint8_t>()),
        reinterpret_cast<int8_t*>(k_mxfp8.data_ptr<uint8_t>()),
        reinterpret_cast<__nv_fp8_e4m3*>(v_mxfp8.data_ptr<uint8_t>()),
        reinterpret_cast<half*>(q_fp16.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_fp16.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_fp16.data_ptr<at::Half>()), nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        block_ids.data_ptr<int32_t>(),
        mxfp8_block_counts.data_ptr<int32_t>(),
        block_ids.data_ptr<int32_t>(), fp16_block_counts.data_ptr<int32_t>(),
        q_mxfp8_scale.data_ptr<uint8_t>(),
        k_mxfp8_scale.data_ptr<uint8_t>(),
        v_mxfp8_scale.data_ptr<uint8_t>(),
        valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
        static_cast<uint32_t>(q_fp16.size(0)),
        static_cast<uint32_t>(q_fp16.size(2)),
        static_cast<uint32_t>(k_fp16.size(2)),
        static_cast<uint32_t>(v_mxfp8.size(3)),
        static_cast<uint32_t>(q_fp16.size(1)),
        static_cast<uint32_t>(k_fp16.size(1)),
        static_cast<float>(softmax_scale), stream);
  } else if constexpr (QueryBlock == 64) {
    auto launcher = active_fp16
        ? launch_mixed_attention_sm120_q64<128, true, true, false>
        : launch_mixed_attention_sm120_q64<128, true, false, false>;
    launcher(
        reinterpret_cast<int8_t*>(q_mxfp8.data_ptr<uint8_t>()),
        reinterpret_cast<int8_t*>(k_mxfp8.data_ptr<uint8_t>()),
        reinterpret_cast<__nv_fp8_e4m3*>(v_mxfp8.data_ptr<uint8_t>()),
        reinterpret_cast<half*>(q_fp16.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_fp16.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_fp16.data_ptr<at::Half>()), nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        block_ids.data_ptr<int32_t>(),
        mxfp8_block_counts.data_ptr<int32_t>(),
        block_ids.data_ptr<int32_t>(),
        fp16_block_counts.data_ptr<int32_t>(),
        q_mxfp8_scale.data_ptr<uint8_t>(),
        k_mxfp8_scale.data_ptr<uint8_t>(),
        v_mxfp8_scale.data_ptr<uint8_t>(),
        valid_k_counts.data_ptr<int32_t>(), nullptr,
        static_cast<uint32_t>(fp16_prefix_blocks),
        static_cast<uint32_t>(q_fp16.size(0)),
        static_cast<uint32_t>(q_fp16.size(2)),
        static_cast<uint32_t>(k_fp16.size(2)),
        static_cast<uint32_t>(v_mxfp8.size(3)),
        static_cast<uint32_t>(q_fp16.size(1)),
        static_cast<uint32_t>(k_fp16.size(1)),
        static_cast<float>(softmax_scale), stream);
  } else {
    auto launcher = active_fp16
        ? launch_mixed_attention_sm120_q128_mxfp8<128, true, true, false>
        : launch_mixed_attention_sm120_q128_mxfp8<128, true, false, false>;
    launcher(
      reinterpret_cast<int8_t*>(q_mxfp8.data_ptr<uint8_t>()),
      reinterpret_cast<int8_t*>(k_mxfp8.data_ptr<uint8_t>()),
      reinterpret_cast<__nv_fp8_e4m3*>(v_mxfp8.data_ptr<uint8_t>()),
      reinterpret_cast<half*>(q_fp16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(k_fp16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(v_fp16.data_ptr<at::Half>()), nullptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      block_ids.data_ptr<int32_t>(),
      mxfp8_block_counts.data_ptr<int32_t>(),
      block_ids.data_ptr<int32_t>(),
      fp16_block_counts.data_ptr<int32_t>(),
      q_mxfp8_scale.data_ptr<uint8_t>(),
      k_mxfp8_scale.data_ptr<uint8_t>(),
      v_mxfp8_scale.data_ptr<uint8_t>(),
      valid_k_counts.data_ptr<int32_t>(), nullptr,
      static_cast<uint32_t>(fp16_prefix_blocks),
      static_cast<uint32_t>(q_fp16.size(0)),
      static_cast<uint32_t>(q_fp16.size(2)),
      static_cast<uint32_t>(k_fp16.size(2)),
      static_cast<uint32_t>(v_mxfp8.size(3)),
      static_cast<uint32_t>(q_fp16.size(1)),
      static_cast<uint32_t>(k_fp16.size(1)),
      static_cast<float>(softmax_scale), stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

std::tuple<torch::Tensor, torch::Tensor> sm120_q64_mxfp8_attention_forward(
    torch::Tensor q_mxfp8, torch::Tensor q_mxfp8_scale,
    torch::Tensor k_mxfp8, torch::Tensor k_mxfp8_scale,
    torch::Tensor v_mxfp8, torch::Tensor v_mxfp8_scale,
    torch::Tensor q_fp16, torch::Tensor k_fp16, torch::Tensor v_fp16,
    torch::Tensor block_ids, torch::Tensor mxfp8_block_counts,
    torch::Tensor fp16_block_counts, torch::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16) {
  return mxfp8_attention_forward<64>(
      q_mxfp8, q_mxfp8_scale, k_mxfp8, k_mxfp8_scale, v_mxfp8,
      v_mxfp8_scale, q_fp16, k_fp16, v_fp16, block_ids,
      mxfp8_block_counts, fp16_block_counts, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16);
}

std::tuple<torch::Tensor, torch::Tensor> sm120_q128_mxfp8_attention_forward(
    torch::Tensor q_mxfp8, torch::Tensor q_mxfp8_scale,
    torch::Tensor k_mxfp8, torch::Tensor k_mxfp8_scale,
    torch::Tensor v_mxfp8, torch::Tensor v_mxfp8_scale,
    torch::Tensor q_fp16, torch::Tensor k_fp16, torch::Tensor v_fp16,
    torch::Tensor block_ids, torch::Tensor mxfp8_block_counts,
    torch::Tensor fp16_block_counts, torch::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16) {
  return mxfp8_attention_forward<128>(
      q_mxfp8, q_mxfp8_scale, k_mxfp8, k_mxfp8_scale, v_mxfp8,
      v_mxfp8_scale, q_fp16, k_fp16, v_fp16, block_ids,
      mxfp8_block_counts, fp16_block_counts, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16);
}

std::tuple<torch::Tensor, torch::Tensor>
sm120_q128_mxfp8_compact_attention_forward(
    torch::Tensor q_mxfp8, torch::Tensor q_mxfp8_scale,
    torch::Tensor k_mxfp8, torch::Tensor k_mxfp8_scale,
    torch::Tensor v_mxfp8, torch::Tensor v_mxfp8_scale,
    torch::Tensor q_fp16, torch::Tensor k_fp16, torch::Tensor v_fp16,
    torch::Tensor block_ids, torch::Tensor mxfp8_block_counts,
    torch::Tensor fp16_block_counts, torch::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16) {
  return mxfp8_attention_forward<128, true>(
      q_mxfp8, q_mxfp8_scale, k_mxfp8, k_mxfp8_scale, v_mxfp8,
      v_mxfp8_scale, q_fp16, k_fp16, v_fp16, block_ids,
      mxfp8_block_counts, fp16_block_counts, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16);
}

template <uint32_t QueryBlock>
std::tuple<torch::Tensor, torch::Tensor> nvfp4_attention_forward(
    torch::Tensor q_nvfp4, torch::Tensor q_nvfp4_scale,
    torch::Tensor k_nvfp4, torch::Tensor k_nvfp4_scale,
    torch::Tensor v_nvfp4, torch::Tensor v_nvfp4_scale,
    torch::Tensor q_fp16, torch::Tensor k_fp16, torch::Tensor v_fp16,
    torch::Tensor block_ids, torch::Tensor nvfp4_block_counts,
    torch::Tensor fp16_block_counts, torch::Tensor valid_k_counts,
    torch::Tensor q_global_scale, torch::Tensor k_global_scale,
    torch::Tensor v_global_scale, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q_nvfp4, "q_nvfp4"),
           std::pair<const torch::Tensor*, const char*>(&q_nvfp4_scale, "q_nvfp4_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_nvfp4, "k_nvfp4"),
           std::pair<const torch::Tensor*, const char*>(&k_nvfp4_scale, "k_nvfp4_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_nvfp4, "v_nvfp4"),
           std::pair<const torch::Tensor*, const char*>(&v_nvfp4_scale, "v_nvfp4_scale"),
           std::pair<const torch::Tensor*, const char*>(&q_fp16, "q_fp16"),
           std::pair<const torch::Tensor*, const char*>(&k_fp16, "k_fp16"),
           std::pair<const torch::Tensor*, const char*>(&v_fp16, "v_fp16"),
           std::pair<const torch::Tensor*, const char*>(&block_ids, "block_ids"),
           std::pair<const torch::Tensor*, const char*>(&nvfp4_block_counts, "nvfp4_block_counts"),
           std::pair<const torch::Tensor*, const char*>(&fp16_block_counts, "fp16_block_counts"),
           std::pair<const torch::Tensor*, const char*>(&valid_k_counts, "valid_k_counts"),
           std::pair<const torch::Tensor*, const char*>(&q_global_scale, "q_global_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_global_scale, "k_global_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_global_scale, "v_global_scale")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q_fp16, item.second);
  }
  TORCH_CHECK(
      q_fp16.scalar_type() == at::ScalarType::Half &&
          k_fp16.scalar_type() == at::ScalarType::Half &&
          v_fp16.scalar_type() == at::ScalarType::Half,
      "FP16 operands must be FP16");
  TORCH_CHECK(
      q_fp16.dim() == 4 && q_fp16.size(3) == 128 &&
          q_fp16.size(2) > 0 && q_fp16.size(2) % QueryBlock == 0,
      "Q must have shape [B,Hq,Q,128] with Q divisible by query block");
  TORCH_CHECK(
      k_fp16.dim() == 4 && k_fp16.sizes() == v_fp16.sizes() &&
          k_fp16.size(0) == q_fp16.size(0) &&
          k_fp16.size(2) > 0 && k_fp16.size(2) % 64 == 0 &&
          k_fp16.size(3) == 128 && q_fp16.size(1) % k_fp16.size(1) == 0,
      "K/V must have matching [B,Hkv,K,128] shapes with K divisible by 64");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q_nvfp4, "q_nvfp4"),
           std::pair<const torch::Tensor*, const char*>(&q_nvfp4_scale, "q_nvfp4_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_nvfp4, "k_nvfp4"),
           std::pair<const torch::Tensor*, const char*>(&k_nvfp4_scale, "k_nvfp4_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_nvfp4, "v_nvfp4"),
           std::pair<const torch::Tensor*, const char*>(&v_nvfp4_scale, "v_nvfp4_scale")}) {
    TORCH_CHECK(item.first->scalar_type() == at::ScalarType::Byte,
                item.second, " must be uint8");
  }
  TORCH_CHECK(
      q_nvfp4.sizes() == torch::IntArrayRef(
          {q_fp16.size(0), q_fp16.size(1), q_fp16.size(2), 64}) &&
          q_nvfp4_scale.sizes() == torch::IntArrayRef(
              {q_fp16.size(0), q_fp16.size(1), q_fp16.size(2), 8}),
      "invalid NVFP4 Q shapes");
  TORCH_CHECK(
      k_nvfp4.sizes() == torch::IntArrayRef(
          {k_fp16.size(0), k_fp16.size(1), k_fp16.size(2), 64}) &&
          k_nvfp4_scale.sizes() == torch::IntArrayRef(
              {k_fp16.size(0), k_fp16.size(1), k_fp16.size(2), 8}),
      "invalid NVFP4 K shapes");
  const int64_t query_blocks = q_fp16.size(2) / QueryBlock;
  const int64_t key_blocks = k_fp16.size(2) / 64;
  TORCH_CHECK(
      v_nvfp4.sizes() == torch::IntArrayRef(
          {v_fp16.size(0), v_fp16.size(1), 128, v_fp16.size(2) / 2}) &&
          v_nvfp4_scale.sizes() == torch::IntArrayRef(
              {v_fp16.size(0), v_fp16.size(1), key_blocks, 512}),
      "invalid NVFP4 V consumer shapes");
  TORCH_CHECK(
      block_ids.scalar_type() == at::ScalarType::Int &&
          nvfp4_block_counts.scalar_type() == at::ScalarType::Int &&
          fp16_block_counts.scalar_type() == at::ScalarType::Int &&
          valid_k_counts.scalar_type() == at::ScalarType::Int,
      "route metadata must be int32");
  TORCH_CHECK(
      block_ids.sizes() == torch::IntArrayRef(
          {q_fp16.size(0), q_fp16.size(1), query_blocks, key_blocks}) &&
          nvfp4_block_counts.sizes() == torch::IntArrayRef(
              {q_fp16.size(0), q_fp16.size(1), query_blocks}) &&
          (active_fp16
               ? fp16_block_counts.sizes() == torch::IntArrayRef(
                     {q_fp16.size(0), q_fp16.size(1), query_blocks})
               : fp16_block_counts.numel() == 0) &&
          valid_k_counts.sizes() == torch::IntArrayRef(
              {q_fp16.size(0), key_blocks}),
      "invalid route metadata shapes");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q_global_scale, "q_global_scale"),
           std::pair<const torch::Tensor*, const char*>(&k_global_scale, "k_global_scale"),
           std::pair<const torch::Tensor*, const char*>(&v_global_scale, "v_global_scale")}) {
    TORCH_CHECK(
        item.first->scalar_type() == at::ScalarType::Float &&
            item.first->numel() == 1,
        item.second, " must be one FP32 scalar");
  }
  TORCH_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks,
      "fp16_prefix_blocks must be in [0,K/64]");
  TORCH_CHECK(
      active_fp16 || fp16_prefix_blocks == 0,
      "inactive FP16 requires zero prefix stages");
  TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
              "softmax_scale must be finite and positive");

  auto output = torch::empty_like(q_fp16);
  auto lse = torch::empty(
      {0},
      q_fp16.options().dtype(at::ScalarType::Float));
  c10::cuda::CUDAGuard device_guard(q_fp16.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_fp16.get_device());
  auto launch_nvfp4 = [&](auto launcher) {
    launcher(
      reinterpret_cast<int8_t*>(q_nvfp4.data_ptr<uint8_t>()),
      reinterpret_cast<int8_t*>(k_nvfp4.data_ptr<uint8_t>()),
      reinterpret_cast<__nv_fp8_e4m3*>(v_nvfp4.data_ptr<uint8_t>()),
      reinterpret_cast<half*>(q_fp16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(k_fp16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(v_fp16.data_ptr<at::Half>()), nullptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      block_ids.data_ptr<int32_t>(), nvfp4_block_counts.data_ptr<int32_t>(),
      block_ids.data_ptr<int32_t>(), fp16_block_counts.data_ptr<int32_t>(),
      q_nvfp4_scale.data_ptr<uint8_t>(),
      k_nvfp4_scale.data_ptr<uint8_t>(),
      v_nvfp4_scale.data_ptr<uint8_t>(),
      q_global_scale.data_ptr<float>(), k_global_scale.data_ptr<float>(),
      v_global_scale.data_ptr<float>(), valid_k_counts.data_ptr<int32_t>(),
      nullptr, static_cast<uint32_t>(fp16_prefix_blocks),
      static_cast<uint32_t>(q_fp16.size(0)),
      static_cast<uint32_t>(q_fp16.size(2)),
      static_cast<uint32_t>(k_fp16.size(2)),
      static_cast<uint32_t>(k_fp16.size(2)),
      static_cast<uint32_t>(q_fp16.size(1)),
      static_cast<uint32_t>(k_fp16.size(1)),
      static_cast<float>(softmax_scale), stream);
  };
  if constexpr (QueryBlock == 64) {
    auto launcher = active_fp16
        ? launch_mixed_attention_sm120_q64_nvfp4<128, true, true, false>
        : launch_mixed_attention_sm120_q64_nvfp4<128, true, false, false>;
    launch_nvfp4(launcher);
  } else {
    auto launcher = active_fp16
        ? launch_mixed_attention_sm120_q128_nvfp4<128, true, true, false>
        : launch_mixed_attention_sm120_q128_nvfp4<128, true, false, false>;
    launch_nvfp4(launcher);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

#define MPA_DEFINE_NVFP4_FORWARD(name, query_block) \
std::tuple<torch::Tensor, torch::Tensor> name( \
    torch::Tensor q_nvfp4, torch::Tensor q_nvfp4_scale, \
    torch::Tensor k_nvfp4, torch::Tensor k_nvfp4_scale, \
    torch::Tensor v_nvfp4, torch::Tensor v_nvfp4_scale, \
    torch::Tensor q_fp16, torch::Tensor k_fp16, torch::Tensor v_fp16, \
    torch::Tensor block_ids, torch::Tensor nvfp4_block_counts, \
    torch::Tensor fp16_block_counts, torch::Tensor valid_k_counts, \
    torch::Tensor q_global_scale, torch::Tensor k_global_scale, \
    torch::Tensor v_global_scale, int64_t fp16_prefix_blocks, \
    double softmax_scale, bool active_fp16) { \
  return nvfp4_attention_forward<query_block>( \
      q_nvfp4, q_nvfp4_scale, k_nvfp4, k_nvfp4_scale, v_nvfp4, \
      v_nvfp4_scale, q_fp16, k_fp16, v_fp16, block_ids, \
      nvfp4_block_counts, fp16_block_counts, valid_k_counts, \
      q_global_scale, k_global_scale, v_global_scale, fp16_prefix_blocks, \
      softmax_scale, active_fp16); \
}

MPA_DEFINE_NVFP4_FORWARD(sm120_q64_nvfp4_attention_forward, 64)
MPA_DEFINE_NVFP4_FORWARD(sm120_q128_nvfp4_attention_forward, 128)
#undef MPA_DEFINE_NVFP4_FORWARD

template <uint32_t QueryBlock, bool MiddleInt8>
std::tuple<torch::Tensor, torch::Tensor> three_phase_forward(
    torch::Tensor q4, torch::Tensor q4_scale,
    torch::Tensor k4, torch::Tensor k4_scale,
    torch::Tensor v4, torch::Tensor v4_scale,
    torch::Tensor q8, torch::Tensor q8_scale,
    torch::Tensor k8, torch::Tensor k8_scale,
    torch::Tensor v8, torch::Tensor v8_scale,
    torch::Tensor q16, torch::Tensor k16, torch::Tensor v16,
    torch::Tensor block_ids, torch::Tensor nvfp4_counts,
    torch::Tensor middle_counts, torch::Tensor fp16_counts,
    torch::Tensor valid_k_counts, torch::Tensor q_global_scale,
    torch::Tensor k_global_scale, torch::Tensor v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&q4, "q4"),
           {&q4_scale, "q4_scale"}, {&k4, "k4"},
           {&k4_scale, "k4_scale"}, {&v4, "v4"},
           {&v4_scale, "v4_scale"}, {&q8, "q8"},
           {&q8_scale, "q8_scale"}, {&k8, "k8"},
           {&k8_scale, "k8_scale"}, {&v8, "v8"},
           {&v8_scale, "v8_scale"}, {&q16, "q16"},
           {&k16, "k16"}, {&v16, "v16"},
           {&block_ids, "block_ids"}, {&nvfp4_counts, "nvfp4_counts"},
           {&middle_counts, "middle_counts"}, {&fp16_counts, "fp16_counts"},
           {&valid_k_counts, "valid_k_counts"},
           {&q_global_scale, "q_global_scale"},
           {&k_global_scale, "k_global_scale"},
           {&v_global_scale, "v_global_scale"}}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q16, item.second);
  }
  TORCH_CHECK(
      q16.scalar_type() == at::ScalarType::Half &&
          k16.scalar_type() == at::ScalarType::Half &&
          v16.scalar_type() == at::ScalarType::Half &&
          q16.dim() == 4 && k16.dim() == 4 && k16.sizes() == v16.sizes() &&
          q16.size(0) == k16.size(0) && q16.size(2) > 0 &&
          q16.size(2) % QueryBlock == 0 && k16.size(2) > 0 &&
          k16.size(2) % 64 == 0 && q16.size(3) == 128 &&
          k16.size(3) == 128 && q16.size(1) % k16.size(1) == 0,
      "FP16 operands must be compatible [B,H,Q/K,128] tensors");
  const int64_t query_blocks = q16.size(2) / QueryBlock;
  const int64_t key_blocks = k16.size(2) / 64;
  const std::array<int64_t, 3> count_dims = {
      q16.size(0), q16.size(1), query_blocks};
  const torch::IntArrayRef count_shape(count_dims);
  TORCH_CHECK(
      block_ids.scalar_type() == at::ScalarType::Int &&
          block_ids.sizes() == torch::IntArrayRef(
              {q16.size(0), q16.size(1), query_blocks, key_blocks}) &&
          nvfp4_counts.scalar_type() == at::ScalarType::Int &&
          nvfp4_counts.sizes() == count_shape &&
          middle_counts.scalar_type() == at::ScalarType::Int &&
          middle_counts.sizes() == count_shape &&
          fp16_counts.scalar_type() == at::ScalarType::Int &&
          (active_fp16
               ? fp16_counts.sizes() == count_shape
               : fp16_counts.numel() == 0) &&
          valid_k_counts.scalar_type() == at::ScalarType::Int &&
          valid_k_counts.sizes() == torch::IntArrayRef(
              {q16.size(0), key_blocks}),
      "invalid compact three-phase route metadata");
  TORCH_CHECK(
      q4.scalar_type() == at::ScalarType::Byte &&
          k4.scalar_type() == at::ScalarType::Byte &&
          v4.scalar_type() == at::ScalarType::Byte &&
          q4_scale.scalar_type() == at::ScalarType::Byte &&
          k4_scale.scalar_type() == at::ScalarType::Byte &&
          v4_scale.scalar_type() == at::ScalarType::Byte &&
          q4.sizes() == torch::IntArrayRef(
              {q16.size(0), q16.size(1), q16.size(2), 64}) &&
          q4_scale.sizes() == torch::IntArrayRef(
              {q16.size(0), q16.size(1), q16.size(2), 8}) &&
          k4.sizes() == torch::IntArrayRef(
              {k16.size(0), k16.size(1), k16.size(2), 64}) &&
          k4_scale.sizes() == torch::IntArrayRef(
              {k16.size(0), k16.size(1), k16.size(2), 8}) &&
          v4.sizes() == torch::IntArrayRef(
              {v16.size(0), v16.size(1), 128, v16.size(2) / 2}) &&
          v4_scale.sizes() == torch::IntArrayRef(
              {v16.size(0), v16.size(1), key_blocks, 512}),
      "invalid NVFP4 operand shapes or dtypes");
  for (const auto* scale : {&q_global_scale, &k_global_scale, &v_global_scale}) {
    TORCH_CHECK(
        scale->scalar_type() == at::ScalarType::Float && scale->numel() == 1,
        "NVFP4 tensor-global scales must be FP32 scalars");
  }
  if constexpr (MiddleInt8) {
    TORCH_CHECK(
        q8.scalar_type() == at::ScalarType::Char &&
            k8.scalar_type() == at::ScalarType::Char &&
            v8.scalar_type() == at::ScalarType::Float8_e4m3fn &&
            q8.sizes() == q16.sizes() && k8.sizes() == k16.sizes() &&
            q8_scale.scalar_type() == at::ScalarType::Float &&
            q8_scale.sizes() == count_shape &&
            k8_scale.scalar_type() == at::ScalarType::Float &&
            k8_scale.sizes() == torch::IntArrayRef(
                {q16.size(0), k16.size(1), key_blocks}) &&
            v8_scale.scalar_type() == at::ScalarType::Float &&
            v8_scale.sizes() == torch::IntArrayRef(
                {q16.size(0), k16.size(1), 128}),
        "invalid INT8/E4M3 middle-phase operands");
  } else {
    TORCH_CHECK(
        q8.scalar_type() == at::ScalarType::Byte &&
            k8.scalar_type() == at::ScalarType::Byte &&
            v8.scalar_type() == at::ScalarType::Byte &&
            q8.sizes() == q16.sizes() && k8.sizes() == k16.sizes() &&
            q8_scale.scalar_type() == at::ScalarType::Byte &&
            q8_scale.sizes() == torch::IntArrayRef(
                {q16.size(0), q16.size(1), q16.size(2), 4}) &&
            k8_scale.scalar_type() == at::ScalarType::Byte &&
            k8_scale.sizes() == torch::IntArrayRef(
                {k16.size(0), k16.size(1), k16.size(2), 4}) &&
            v8_scale.scalar_type() == at::ScalarType::Byte &&
            v8_scale.sizes() == torch::IntArrayRef(
                {k16.size(0), k16.size(1), key_blocks, 256}),
        "invalid MXFP8 middle-phase operands");
  }
  TORCH_CHECK(
      v8.dim() == 4 && v8.size(0) == q16.size(0) &&
          v8.size(1) == k16.size(1) && v8.size(2) == 128 &&
          v8.size(3) >= k16.size(2) && v8.size(3) % 64 == 0,
      "middle V must have shape [B,Hkv,128,padded_K]");
  TORCH_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks &&
          std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "invalid prefix count or softmax scale");
  TORCH_CHECK(
      active_fp16 || fp16_prefix_blocks == 0,
      "inactive FP16 requires zero prefix stages");
  auto output = torch::empty_like(q16);
  auto lse = torch::empty(
      {0},
      q16.options().dtype(at::ScalarType::Float));
  c10::cuda::CUDAGuard device_guard(q16.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q16.get_device());
#define MPA_STACK_COMMON_ARGS \
      reinterpret_cast<half*>(q16.data_ptr<at::Half>()), \
      reinterpret_cast<half*>(k16.data_ptr<at::Half>()), \
      reinterpret_cast<half*>(v16.data_ptr<at::Half>()), nullptr, \
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), \
      block_ids.data_ptr<int32_t>(), nvfp4_counts.data_ptr<int32_t>(), \
      block_ids.data_ptr<int32_t>(), fp16_counts.data_ptr<int32_t>()
#define MPA_STACK_NV_ARGS \
      reinterpret_cast<int8_t*>(q4.data_ptr<uint8_t>()), \
      reinterpret_cast<int8_t*>(k4.data_ptr<uint8_t>()), \
      reinterpret_cast<__nv_fp8_e4m3*>(v4.data_ptr<uint8_t>()), \
      q4_scale.data_ptr<uint8_t>(), k4_scale.data_ptr<uint8_t>(), \
      v4_scale.data_ptr<uint8_t>(), middle_counts.data_ptr<int32_t>(), \
      q_global_scale.data_ptr<float>(), k_global_scale.data_ptr<float>(), \
      v_global_scale.data_ptr<float>(), valid_k_counts.data_ptr<int32_t>(), \
      nullptr, static_cast<uint32_t>(fp16_prefix_blocks), \
      static_cast<uint32_t>(q16.size(0)), static_cast<uint32_t>(q16.size(2)), \
      static_cast<uint32_t>(k16.size(2)), static_cast<uint32_t>(v8.size(3)), \
      static_cast<uint32_t>(q16.size(1)), static_cast<uint32_t>(k16.size(1)), \
      static_cast<float>(softmax_scale), stream
  if constexpr (MiddleInt8) {
    auto launch_int8 = [&](auto launcher) {
      launcher(
          q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
          reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
          MPA_STACK_COMMON_ARGS,
          q8_scale.data_ptr<float>(), k8_scale.data_ptr<float>(),
          v8_scale.data_ptr<float>(), MPA_STACK_NV_ARGS);
    };
    if constexpr (QueryBlock == 64) {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q64_nv_int8_fp16<128, true, true, false>
          : launch_mixed_attention_sm120_q64_nv_int8_fp16<128, true, false, false>;
      launch_int8(launcher);
    } else {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q128_nv_int8_fp16<128, true, true, false>
          : launch_mixed_attention_sm120_q128_nv_int8_fp16<128, true, false, false>;
      launch_int8(launcher);
    }
  } else {
    auto launch_mx = [&](auto launcher) {
      launcher(
          reinterpret_cast<int8_t*>(q8.data_ptr<uint8_t>()),
          reinterpret_cast<int8_t*>(k8.data_ptr<uint8_t>()),
          reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr<uint8_t>()),
          MPA_STACK_COMMON_ARGS,
          q8_scale.data_ptr<uint8_t>(), k8_scale.data_ptr<uint8_t>(),
          v8_scale.data_ptr<uint8_t>(), MPA_STACK_NV_ARGS);
    };
    if constexpr (QueryBlock == 64) {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q64_nv_mx_fp16<128, true, true, false>
          : launch_mixed_attention_sm120_q64_nv_mx_fp16<128, true, false, false>;
      launch_mx(launcher);
    } else {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q128_nv_mx_fp16<128, true, true, false>
          : launch_mixed_attention_sm120_q128_nv_mx_fp16<128, true, false, false>;
      launch_mx(launcher);
    }
  }
#undef MPA_STACK_NV_ARGS
#undef MPA_STACK_COMMON_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

#define MPA_DEFINE_NV_INT8_FORWARD(name, query_block) \
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
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16) { \
  return three_phase_forward<query_block, true>( \
      q4, q4_scale, k4, k4_scale, v4, v4_scale, q8, q8_scale, k8, \
      k8_scale, v8, v8_scale, q16, k16, v16, block_ids, nvfp4_counts, \
      middle_counts, fp16_counts, valid_k_counts, q_global_scale, \
      k_global_scale, v_global_scale, fp16_prefix_blocks, softmax_scale, \
      active_fp16); \
}

MPA_DEFINE_NV_INT8_FORWARD(sm120_q64_nv_int8_fp16_attention_forward, 64)
MPA_DEFINE_NV_INT8_FORWARD(sm120_q128_nv_int8_fp16_attention_forward, 128)
#undef MPA_DEFINE_NV_INT8_FORWARD

#define MPA_DEFINE_NV_MX_FORWARD(name, query_block) \
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
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16) { \
  return three_phase_forward<query_block, false>( \
      q4, q4_scale, k4, k4_scale, v4, v4_scale, q8, q8_scale, k8, \
      k8_scale, v8, v8_scale, q16, k16, v16, block_ids, nvfp4_counts, \
      middle_counts, fp16_counts, valid_k_counts, q_global_scale, \
      k_global_scale, v_global_scale, fp16_prefix_blocks, softmax_scale, \
      active_fp16); \
}

MPA_DEFINE_NV_MX_FORWARD(sm120_q64_nv_mx_fp16_attention_forward, 64)
MPA_DEFINE_NV_MX_FORWARD(sm120_q128_nv_mx_fp16_attention_forward, 128)
#undef MPA_DEFINE_NV_MX_FORWARD
