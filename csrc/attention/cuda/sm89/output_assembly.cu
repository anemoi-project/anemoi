/*
 * Copyright 2026 Mixed Attention Project Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fast, allocation-inclusive final output assembly for the EVG integration.
 * This translation unit deliberately inherits the attention extension's
 * --use_fast_math policy.  Numerical acceptance is therefore defined against
 * the eager construction by end-to-end error metrics, not bitwise identity.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "api.h"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
// This caps queued grid work per SM; it is not a claim that 16 CTAs can be
// simultaneously resident.
constexpr int kQueuedBlocksPerSm = 16;

inline int64_t checked_nonnegative_product(
    int64_t lhs, int64_t rhs, const char* description) {
  TORCH_CHECK(lhs >= 0 && rhs >= 0, description, " factors must be nonnegative");
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  TORCH_CHECK(
      lhs <= std::numeric_limits<int64_t>::max() / rhs,
      description, " exceeds int64 range");
  return lhs * rhs;
}

inline void check_cuda_contiguous(
    const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be a defined tensor");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_bf16_rank4(
    const torch::Tensor& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::BFloat16,
      name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.dim() == 4, name, " must be rank-four");
}

inline void check_fp32_bhs(
    const torch::Tensor& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Float,
      name, " must have dtype torch.float32");
  TORCH_CHECK(tensor.dim() == 3, name, " must have shape [B,H,S_video]");
}

__device__ __forceinline__ float stable_logaddexp(float left, float right) {
  // This is the same stable construction used by torch.logaddexp.  The equal
  // infinity guard prevents inf-inf from turning two identical infinities into
  // NaN.  Device exp/log1p follow this extension's accepted fast-math policy.
  if (isinf(left) && left == right) {
    return left;
  }
  const float maximum = left > right ? left : right;
  return maximum + log1pf(expf(-fabsf(left - right)));
}

__global__ __launch_bounds__(kThreads) void assemble_video_text_output_kernel(
    const __nv_bfloat16* __restrict__ video_output_bhsd,
    const float* __restrict__ video_lse_bhs,
    const __nv_bfloat16* __restrict__ visual_text_output_bshd,
    const float* __restrict__ visual_text_lse_bhs,
    const __nv_bfloat16* __restrict__ text_output_bshd,
    const bool* __restrict__ text_mask,
    __nv_bfloat16* __restrict__ output_bshd,
    int64_t visual_length,
    int64_t text_length,
    int64_t heads,
    int64_t dimension,
    bool has_text_mask,
    int64_t text_mask_batch_stride,
    int64_t text_mask_sequence_stride,
    int64_t output_rows) {
  const int64_t sequence_length = visual_length + text_length;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       row < output_rows;
       row += static_cast<int64_t>(gridDim.x) * kWarpsPerBlock) {
    const int64_t head = row % heads;
    const int64_t sequence = (row / heads) % sequence_length;
    const int64_t batch = row / (heads * sequence_length);
    const int64_t output_offset = row * dimension;

    if (sequence < visual_length) {
      float video_weight = 0.0f;
      float text_weight = 0.0f;
      if (lane == 0) {
        const int64_t lse_offset =
            (batch * heads + head) * visual_length + sequence;
        const float video_lse = video_lse_bhs[lse_offset];
        const float text_lse = visual_text_lse_bhs[lse_offset];
        const float merged_lse = stable_logaddexp(video_lse, text_lse);
        video_weight = expf(video_lse - merged_lse);
        text_weight = expf(text_lse - merged_lse);
      }
      video_weight = __shfl_sync(0xffffffffu, video_weight, 0);
      text_weight = __shfl_sync(0xffffffffu, text_weight, 0);

      const int64_t video_offset =
          ((batch * heads + head) * visual_length + sequence) * dimension;
      const int64_t dense_offset =
          ((batch * visual_length + sequence) * heads + head) * dimension;
      for (int64_t column = lane; column < dimension;
           column += kWarpSize) {
        const float video_value =
            __bfloat162float(video_output_bhsd[video_offset + column]);
        const float text_value =
            __bfloat162float(visual_text_output_bshd[dense_offset + column]);
        const float video_product = __fmul_rn(video_value, video_weight);
        const float text_product = __fmul_rn(text_value, text_weight);
        output_bshd[output_offset + column] =
            __float2bfloat16_rn(__fadd_rn(video_product, text_product));
      }
    } else {
      const int64_t text_sequence = sequence - visual_length;
      const bool valid = !has_text_mask ||
          text_mask[
              batch * text_mask_batch_stride +
              text_sequence * text_mask_sequence_stride];
      const int64_t text_offset =
          ((batch * text_length + text_sequence) * heads + head) * dimension;
      for (int64_t column = lane; column < dimension;
           column += kWarpSize) {
        // __float2bfloat16_rn(0.0f) is an explicit +0, even when the source
        // contains -0 or NaN at a masked text position.
        output_bshd[output_offset + column] = valid
            ? text_output_bshd[text_offset + column]
            : __float2bfloat16_rn(0.0f);
      }
    }
  }
}

// Fused output boundary. The visual-query softmax has already traversed both
// video and text K/V inside one CTA, so there is no second visual partition or
// LSE state to merge here.  This kernel only converts the native BHSD visual
// prefix to EVG's BSHD layout and appends the existing text-query output.
__global__ __launch_bounds__(kThreads)
void assemble_fused_visual_text_output_kernel(
    const __nv_bfloat16* __restrict__ visual_output_bhsd,
    const __nv_bfloat16* __restrict__ text_output_bshd,
    const bool* __restrict__ text_mask,
    __nv_bfloat16* __restrict__ output_bshd,
    int64_t visual_length,
    int64_t text_length,
    int64_t heads,
    int64_t dimension,
    bool has_text_mask,
    int64_t text_mask_batch_stride,
    int64_t text_mask_sequence_stride,
    int64_t output_rows) {
  const int64_t sequence_length = visual_length + text_length;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       row < output_rows;
       row += static_cast<int64_t>(gridDim.x) * kWarpsPerBlock) {
    const int64_t head = row % heads;
    const int64_t sequence = (row / heads) % sequence_length;
    const int64_t batch = row / (heads * sequence_length);
    const int64_t output_offset = row * dimension;

    if (sequence < visual_length) {
      const int64_t visual_offset =
          ((batch * heads + head) * visual_length + sequence) * dimension;
      for (int64_t column = lane; column < dimension;
           column += kWarpSize) {
        output_bshd[output_offset + column] =
            visual_output_bhsd[visual_offset + column];
      }
    } else {
      const int64_t text_sequence = sequence - visual_length;
      const bool valid = !has_text_mask ||
          text_mask[
              batch * text_mask_batch_stride +
              text_sequence * text_mask_sequence_stride];
      const int64_t text_offset =
          ((batch * text_length + text_sequence) * heads + head) * dimension;
      for (int64_t column = lane; column < dimension;
           column += kWarpSize) {
        output_bshd[output_offset + column] = valid
            ? text_output_bshd[text_offset + column]
            : __float2bfloat16_rn(0.0f);
      }
    }
  }
}

template <typename OutputT>
__global__ __launch_bounds__(kThreads) void assemble_h3_k64_output_kernel(
    const OutputT* __restrict__ prefix_output_bhsd,
    const half* __restrict__ video_output_bhsd,
    const int64_t* __restrict__ video_inverse_indices,
    OutputT* __restrict__ output_bshd,
    int64_t prefix_batch_stride,
    int64_t prefix_head_stride,
    int64_t prefix_sequence_stride,
    int64_t prefix_tokens,
    int64_t video_tokens,
    int64_t video_capacity,
    int64_t heads,
    int64_t dimension,
    int64_t output_rows) {
  const int64_t sequence_tokens = prefix_tokens + video_tokens;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       row < output_rows;
       row += static_cast<int64_t>(gridDim.x) * kWarpsPerBlock) {
    const int64_t head = row % heads;
    const int64_t sequence = (row / heads) % sequence_tokens;
    const int64_t batch = row / (heads * sequence_tokens);
    const int64_t output_offset = row * dimension;
    if (sequence < prefix_tokens) {
      const int64_t source_offset =
          batch * prefix_batch_stride + head * prefix_head_stride +
          sequence * prefix_sequence_stride;
      for (int64_t column = lane; column < dimension; column += kWarpSize) {
        output_bshd[output_offset + column] =
            prefix_output_bhsd[source_offset + column];
      }
    } else {
      const int64_t video_token = sequence - prefix_tokens;
      const int64_t physical_token = video_inverse_indices[video_token];
      const int64_t source_offset =
          ((batch * heads + head) * video_capacity + physical_token) *
          dimension;
      for (int64_t column = lane; column < dimension; column += kWarpSize) {
        if constexpr (std::is_same_v<OutputT, half>) {
          output_bshd[output_offset + column] =
              video_output_bhsd[source_offset + column];
        } else {
          output_bshd[output_offset + column] = __float2bfloat16_rn(
              __half2float(video_output_bhsd[source_offset + column]));
        }
      }
    }
  }
}

}  // namespace

torch::Tensor assemble_h3_k64_output(
    torch::Tensor prefix_output_bhsd,
    torch::Tensor video_output_bhsd_fp16,
    torch::Tensor video_inverse_indices) {
  TORCH_CHECK(
      prefix_output_bhsd.defined() && prefix_output_bhsd.is_cuda(),
      "prefix_output_bhsd must be a defined CUDA tensor");
  check_cuda_contiguous(video_output_bhsd_fp16, "video_output_bhsd_fp16");
  check_cuda_contiguous(video_inverse_indices, "video_inverse_indices");
  TORCH_CHECK(
      prefix_output_bhsd.scalar_type() == at::ScalarType::Half ||
          prefix_output_bhsd.scalar_type() == at::ScalarType::BFloat16,
      "prefix_output_bhsd must be FP16 or BF16");
  TORCH_CHECK(
      video_output_bhsd_fp16.scalar_type() == at::ScalarType::Half,
      "video_output_bhsd_fp16 must be FP16");
  TORCH_CHECK(
      video_inverse_indices.scalar_type() == at::ScalarType::Long &&
          video_inverse_indices.dim() == 1,
      "video_inverse_indices must be int64 [video_tokens]");
  TORCH_CHECK(
      prefix_output_bhsd.device() == video_output_bhsd_fp16.device() &&
          prefix_output_bhsd.device() == video_inverse_indices.device(),
      "H3 output assembly tensors must share one CUDA device");
  TORCH_CHECK(
      prefix_output_bhsd.dim() == 4 && video_output_bhsd_fp16.dim() == 4,
      "H3 output operands must be rank-four BHSD tensors");
  TORCH_CHECK(
      prefix_output_bhsd.stride(0) > 0 &&
          prefix_output_bhsd.stride(1) > 0 &&
          prefix_output_bhsd.stride(2) > 0 &&
          prefix_output_bhsd.stride(3) == 1,
      "prefix_output_bhsd must have positive B/H/S strides and contiguous D");
  const int64_t batch = prefix_output_bhsd.size(0);
  const int64_t heads = prefix_output_bhsd.size(1);
  const int64_t prefix_tokens = prefix_output_bhsd.size(2);
  const int64_t dimension = prefix_output_bhsd.size(3);
  const int64_t video_tokens = video_inverse_indices.numel();
  const int64_t video_capacity = video_output_bhsd_fp16.size(2);
  TORCH_CHECK(
      batch > 0 && heads > 0 && prefix_tokens > 0 && dimension > 0 &&
          video_tokens > 0 && video_capacity >= video_tokens,
      "H3 output assembly dimensions must be positive and capacity-valid");
  TORCH_CHECK(
      video_output_bhsd_fp16.size(0) == batch &&
          video_output_bhsd_fp16.size(1) == heads &&
          video_output_bhsd_fp16.size(3) == dimension,
      "video_output_bhsd_fp16 must share B/H/D with the prefix output");

  c10::cuda::CUDAGuard device_guard(prefix_output_bhsd.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(prefix_output_bhsd.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 9,
      "H3 K64 output assembly requires compute capability 8.9, found sm_",
      properties->major, properties->minor);
  TORCH_CHECK(
      prefix_tokens <= std::numeric_limits<int64_t>::max() - video_tokens,
      "H3 K64 output sequence length overflows int64");
  const int64_t sequence_tokens = prefix_tokens + video_tokens;
  const int64_t batch_sequence = checked_nonnegative_product(
      batch, sequence_tokens, "H3 K64 output batch-sequence rows");
  const int64_t output_rows = checked_nonnegative_product(
      batch_sequence, heads, "H3 K64 output BSH rows");
  checked_nonnegative_product(
      output_rows, dimension, "H3 K64 output elements");
  auto output = torch::empty(
      {batch, sequence_tokens, heads, dimension},
      prefix_output_bhsd.options());
  const int64_t blocks_needed =
      (output_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t queued_grid_cap = checked_nonnegative_product(
      properties->multiProcessorCount, kQueuedBlocksPerSm,
      "H3 K64 output assembly SM grid cap");
  const int64_t grid_x = std::min(blocks_needed, queued_grid_cap);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(prefix_output_bhsd.get_device());
  if (prefix_output_bhsd.scalar_type() == at::ScalarType::Half) {
    assemble_h3_k64_output_kernel<<<grid_x, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(prefix_output_bhsd.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(video_output_bhsd_fp16.data_ptr<at::Half>()),
        video_inverse_indices.data_ptr<int64_t>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        prefix_output_bhsd.stride(0), prefix_output_bhsd.stride(1),
        prefix_output_bhsd.stride(2),
        prefix_tokens, video_tokens, video_capacity, heads, dimension,
        output_rows);
  } else {
    assemble_h3_k64_output_kernel<<<grid_x, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            prefix_output_bhsd.data_ptr<at::BFloat16>()),
        reinterpret_cast<const half*>(video_output_bhsd_fp16.data_ptr<at::Half>()),
        video_inverse_indices.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        prefix_output_bhsd.stride(0), prefix_output_bhsd.stride(1),
        prefix_output_bhsd.stride(2),
        prefix_tokens, video_tokens, video_capacity, heads, dimension,
        output_rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor assemble_video_text_output(
    torch::Tensor video_output_bhsd,
    torch::Tensor video_lse_bhs,
    torch::Tensor visual_text_output_bshd,
    torch::Tensor visual_text_lse_bhs,
    torch::Tensor text_output_bshd,
    torch::Tensor text_mask) {
  check_bf16_rank4(video_output_bhsd, "video_output_bhsd");
  check_fp32_bhs(video_lse_bhs, "video_lse_bhs");
  check_bf16_rank4(visual_text_output_bshd, "visual_text_output_bshd");
  check_fp32_bhs(visual_text_lse_bhs, "visual_text_lse_bhs");
  check_bf16_rank4(text_output_bshd, "text_output_bshd");
  TORCH_CHECK(text_mask.defined(), "text_mask must be a defined tensor");
  TORCH_CHECK(text_mask.is_cuda(), "text_mask must be a CUDA tensor");
  TORCH_CHECK(
      text_mask.scalar_type() == at::ScalarType::Bool,
      "text_mask must have dtype torch.bool");

  const auto device = video_output_bhsd.device();
  const std::array<std::pair<const torch::Tensor*, const char*>, 5>
      same_device_tensors{{
          {&video_lse_bhs, "video_lse_bhs"},
          {&visual_text_output_bshd, "visual_text_output_bshd"},
          {&visual_text_lse_bhs, "visual_text_lse_bhs"},
          {&text_output_bshd, "text_output_bshd"},
          {&text_mask, "text_mask"},
      }};
  for (const auto& item : same_device_tensors) {
    TORCH_CHECK(item.first->device() == device, item.second, " must share one device");
  }

  const int64_t batch_size = video_output_bhsd.size(0);
  const int64_t heads = video_output_bhsd.size(1);
  const int64_t visual_length = video_output_bhsd.size(2);
  const int64_t dimension = video_output_bhsd.size(3);
  TORCH_CHECK(
      batch_size > 0 && heads > 0 && visual_length > 0 && dimension > 0,
      "video_output_bhsd dimensions must be positive");
  TORCH_CHECK(
      video_lse_bhs.sizes() ==
          torch::IntArrayRef({batch_size, heads, visual_length}),
      "video_lse_bhs must have shape [B,H,S_video]");
  TORCH_CHECK(
      visual_text_output_bshd.sizes() ==
          torch::IntArrayRef({batch_size, visual_length, heads, dimension}),
      "visual_text_output_bshd must have shape [B,S_video,H,D]");
  TORCH_CHECK(
      visual_text_lse_bhs.sizes() == video_lse_bhs.sizes(),
      "visual_text_lse_bhs must have shape [B,H,S_video]");
  TORCH_CHECK(
      text_output_bshd.size(0) == batch_size &&
          text_output_bshd.size(2) == heads &&
          text_output_bshd.size(3) == dimension,
      "text_output_bshd must have shape [B,S_text,H,D]");
  const int64_t text_length = text_output_bshd.size(1);
  const bool has_text_mask = text_mask.numel() != 0;
  if (has_text_mask) {
    TORCH_CHECK(
        text_mask.dim() == 2 && text_mask.size(0) == batch_size &&
            text_mask.size(1) == text_length,
        "nonempty text_mask must have shape [B,S_text]");
    TORCH_CHECK(
        text_mask.stride(0) >= 0 && text_mask.stride(1) >= 0,
        "text_mask strides must be nonnegative");
  } else {
    TORCH_CHECK(
        text_mask.dim() == 1 && text_mask.size(0) == 0,
        "empty text_mask sentinel must have shape [0]");
  }

  TORCH_CHECK(
      visual_length <= std::numeric_limits<int64_t>::max() - text_length,
      "S_video+S_text exceeds int64 range");
  const int64_t sequence_length = visual_length + text_length;
  const int64_t batch_sequence = checked_nonnegative_product(
      batch_size, sequence_length, "output B*S");
  const int64_t output_rows = checked_nonnegative_product(
      batch_sequence, heads, "output B*S*H");
  checked_nonnegative_product(output_rows, dimension, "output elements");

  c10::cuda::CUDAGuard device_guard(device);
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(video_output_bhsd.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 9,
      "evg.layers.attention.mpa._cuda_attention output assembly requires compute capability 8.9, found sm_",
      properties->major, properties->minor);

  auto output = torch::empty(
      {batch_size, sequence_length, heads, dimension},
      video_output_bhsd.options());
  const int64_t blocks_needed =
      output_rows / kWarpsPerBlock +
      static_cast<int64_t>(output_rows % kWarpsPerBlock != 0);
  const int64_t queued_grid_cap = checked_nonnegative_product(
      properties->multiProcessorCount, kQueuedBlocksPerSm,
      "output assembly SM grid cap");
  const int64_t grid_x = std::min(blocks_needed, queued_grid_cap);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(video_output_bhsd.get_device());
  assemble_video_text_output_kernel<<<
      static_cast<unsigned int>(grid_x), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          video_output_bhsd.data_ptr<at::BFloat16>()),
      video_lse_bhs.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(
          visual_text_output_bshd.data_ptr<at::BFloat16>()),
      visual_text_lse_bhs.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(
          text_output_bshd.data_ptr<at::BFloat16>()),
      text_mask.data_ptr<bool>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      visual_length, text_length, heads, dimension,
      has_text_mask,
      has_text_mask ? text_mask.stride(0) : 0,
      has_text_mask ? text_mask.stride(1) : 0,
      output_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor assemble_fused_visual_text_output(
    torch::Tensor visual_output_bhsd,
    torch::Tensor text_output_bshd,
    torch::Tensor text_mask) {
  check_bf16_rank4(visual_output_bhsd, "visual_output_bhsd");
  check_bf16_rank4(text_output_bshd, "text_output_bshd");
  TORCH_CHECK(text_mask.defined(), "text_mask must be a defined tensor");
  TORCH_CHECK(text_mask.is_cuda(), "text_mask must be a CUDA tensor");
  TORCH_CHECK(
      text_mask.scalar_type() == at::ScalarType::Bool,
      "text_mask must have dtype torch.bool");

  const auto device = visual_output_bhsd.device();
  TORCH_CHECK(
      text_output_bshd.device() == device,
      "text_output_bshd must share the visual output device");
  TORCH_CHECK(
      text_mask.device() == device,
      "text_mask must share the visual output device");

  const int64_t batch_size = visual_output_bhsd.size(0);
  const int64_t heads = visual_output_bhsd.size(1);
  const int64_t visual_length = visual_output_bhsd.size(2);
  const int64_t dimension = visual_output_bhsd.size(3);
  TORCH_CHECK(
      batch_size > 0 && heads > 0 && visual_length > 0 && dimension > 0,
      "visual_output_bhsd dimensions must be positive");
  TORCH_CHECK(
      text_output_bshd.size(0) == batch_size &&
          text_output_bshd.size(2) == heads &&
          text_output_bshd.size(3) == dimension,
      "text_output_bshd must have shape [B,S_text,H,D]");
  const int64_t text_length = text_output_bshd.size(1);
  const bool has_text_mask = text_mask.numel() != 0;
  if (has_text_mask) {
    TORCH_CHECK(
        text_mask.dim() == 2 && text_mask.size(0) == batch_size &&
            text_mask.size(1) == text_length,
        "nonempty text_mask must have shape [B,S_text]");
    TORCH_CHECK(
        text_mask.stride(0) >= 0 && text_mask.stride(1) >= 0,
        "text_mask strides must be nonnegative");
  } else {
    TORCH_CHECK(
        text_mask.dim() == 1 && text_mask.size(0) == 0,
        "empty text_mask sentinel must have shape [0]");
  }

  TORCH_CHECK(
      visual_length <= std::numeric_limits<int64_t>::max() - text_length,
      "S_video+S_text exceeds int64 range");
  const int64_t sequence_length = visual_length + text_length;
  const int64_t batch_sequence = checked_nonnegative_product(
      batch_size, sequence_length, "output B*S");
  const int64_t output_rows = checked_nonnegative_product(
      batch_sequence, heads, "output B*S*H");
  checked_nonnegative_product(output_rows, dimension, "output elements");

  c10::cuda::CUDAGuard device_guard(device);
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(visual_output_bhsd.get_device());
  TORCH_CHECK(
      properties->major == 8 && properties->minor == 9,
      "evg.layers.attention.mpa._cuda_attention fused output assembly requires compute capability 8.9, found sm_",
      properties->major, properties->minor);

  auto output = torch::empty(
      {batch_size, sequence_length, heads, dimension},
      visual_output_bhsd.options());
  const int64_t blocks_needed =
      output_rows / kWarpsPerBlock +
      static_cast<int64_t>(output_rows % kWarpsPerBlock != 0);
  const int64_t queued_grid_cap = checked_nonnegative_product(
      properties->multiProcessorCount, kQueuedBlocksPerSm,
      "fused output assembly SM grid cap");
  const int64_t grid_x = std::min(blocks_needed, queued_grid_cap);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(visual_output_bhsd.get_device());
  assemble_fused_visual_text_output_kernel<<<
      static_cast<unsigned int>(grid_x), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          visual_output_bhsd.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          text_output_bshd.data_ptr<at::BFloat16>()),
      text_mask.data_ptr<bool>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      visual_length, text_length, heads, dimension,
      has_text_mask,
      has_text_mask ? text_mask.stride(0) : 0,
      has_text_mask ? text_mask.stride(1) : 0,
      output_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
