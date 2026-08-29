/*
 * Copyright 2026 Mixed Attention Project Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Project-owned raw-FP16/BF16 raster preprocessing boundary.  It converts
 * video operands from frame-major raster order [B,H,F*Y*X,D] to the logical
 * FP16 8x16 patch sequence [B,H,R*128,D], where
 * R=F*ceil(Y/8)*ceil(X/16).  BF16 is narrowed to FP16 before storage and every
 * virtual edge slot is written as the exact +0 FP16 bit pattern.
 *
 * This increment intentionally does not quantize Q/K or transform V.  The
 * packed operands preserve raw K for the FP16 rescue phase and provide the
 * explicit validity padding required by the later raster-aware quantizers.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include "api.h"
#include "../../../common/raw_bhsd_layout.h"
#include "../common/execution_device.cuh"
#include "../common/packed_raster_layout.cuh"

namespace mpa::attention {
namespace {

constexpr int32_t kPackThreads = 256;
constexpr int32_t kHalfElementsPerVector = 8;

int64_t checked_positive_product(
    int64_t lhs,
    int64_t rhs,
    const char* description) {
  TORCH_CHECK(lhs > 0 && rhs > 0, description, " factors must be positive");
  TORCH_CHECK(
      lhs <= std::numeric_limits<int64_t>::max() / rhs,
      description,
      " exceeds int64 range");
  return lhs * rhs;
}

void check_raw_hnd(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  mpa::check_supported_raw_bhsd_input_view(tensor, name);
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Half ||
          tensor.scalar_type() == at::ScalarType::BFloat16,
      name,
      " must have dtype torch.float16 or torch.bfloat16");
  TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B,H,F*Y*X,D]");
  TORCH_CHECK(
      tensor.size(0) > 0 && tensor.size(1) > 0 && tensor.size(2) > 0,
      name,
      " batch, head, and token dimensions must be positive");
  TORCH_CHECK(
      tensor.size(3) == 64 || tensor.size(3) == 128,
      name,
      " head dimension must be 64 or 128");
}

union Half2Word {
  half2 value;
  uint32_t bits;
};

template <typename InputT>
__device__ __forceinline__ uint4 load_fp16_vector(
    const InputT* input,
    int64_t offset);

template <>
__device__ __forceinline__ uint4 load_fp16_vector<half>(
    const half* input,
    int64_t offset) {
  // Preserve the accepted FP16 bit-copy path exactly.
  return *reinterpret_cast<const uint4*>(input + offset);
}

template <>
__device__ __forceinline__ uint4 load_fp16_vector<nv_bfloat16>(
    const nv_bfloat16* input,
    int64_t offset) {
  // Vectorize four BF16x2 -> FP16x2 RN conversions. The resulting
  // packed tensor is indistinguishable from an eager raw `.to(float16)`
  // followed by the original raster bit-copy, without a separate cast buffer.
  uint32_t words[4];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const nv_bfloat162 source =
        reinterpret_cast<const nv_bfloat162*>(input + offset)[pair];
    Half2Word narrowed;
    narrowed.value = __float22half2_rn(__bfloat1622float2(source));
    words[pair] = narrowed.bits;
  }
  return make_uint4(words[0], words[1], words[2], words[3]);
}

Raster8x16Layout checked_layout(
    int64_t frames,
    int64_t height,
    int64_t width) {
  TORCH_CHECK(frames > 0, "frames must be positive");
  TORCH_CHECK(height > 0, "height must be positive");
  TORCH_CHECK(width > 0, "width must be positive");
  TORCH_CHECK(
      height <= std::numeric_limits<int64_t>::max() -
              (kRasterPatchHeight - 1),
      "height padding arithmetic exceeds int64 range");
  TORCH_CHECK(
      width <= std::numeric_limits<int64_t>::max() -
              (kRasterPatchWidth - 1),
      "width padding arithmetic exceeds int64 range");

  const int64_t patches_h =
      (height + kRasterPatchHeight - 1) / kRasterPatchHeight;
  const int64_t patches_w =
      (width + kRasterPatchWidth - 1) / kRasterPatchWidth;
  const int64_t patches_per_frame = checked_positive_product(
      patches_h, patches_w, "ceil(Y/8)*ceil(X/16)");
  checked_positive_product(
      frames, patches_per_frame, "F*ceil(Y/8)*ceil(X/16)");
  const int64_t frame_tokens =
      checked_positive_product(height, width, "Y*X");
  checked_positive_product(frames, frame_tokens, "F*Y*X");
  return Raster8x16Layout(frames, height, width);
}

void check_raster_and_grid(
    const torch::Tensor& anchor,
    int64_t patch_count,
    int64_t max_heads) {
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(anchor.get_device());
  TORCH_CHECK(
      mpa::attention::sm89_or_sm120_execution_device(properties),
      "Raster preprocessing requires sm89 or sm120, found sm_",
      properties->major,
      properties->minor);
  TORCH_CHECK(
      patch_count <= properties->maxGridSize[0],
      "logical patch count exceeds CUDA grid.x limit");
  TORCH_CHECK(
      max_heads <= properties->maxGridSize[1],
      "head count exceeds CUDA grid.y limit");
  TORCH_CHECK(
      anchor.size(0) <= properties->maxGridSize[2],
      "batch size exceeds CUDA grid.z limit");
}

template <int32_t HeadDim, typename InputT>
__global__ void pack_raster_operand_to_fp16_kernel(
    const InputT* __restrict__ input,
    half* __restrict__ output,
    Raster8x16Layout layout,
    int64_t num_heads,
    int64_t raw_tokens,
    int64_t virtual_tokens,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_token) {
  constexpr int32_t vectors_per_token =
      HeadDim / kHalfElementsPerVector;
  constexpr int32_t vectors_per_patch =
      kRasterPatchTokens * vectors_per_token;

  const int64_t logical_patch = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  for (int32_t vector = threadIdx.x; vector < vectors_per_patch;
       vector += blockDim.x) {
    const int32_t local_token = vector / vectors_per_token;
    const int32_t vector_in_token = vector % vectors_per_token;
    const int32_t element = vector_in_token * kHalfElementsPerVector;
    const RasterTokenAddress address =
        layout.logical_token(logical_patch, local_token);

    // uint4 preserves every valid FP16 payload bit and makes the invalid path
    // an unambiguous positive-zero write, including for originally signed zero
    // inputs at neighboring valid positions.
    uint4 payload = make_uint4(0, 0, 0, 0);
    if (address.valid) {
      const int64_t input_offset = batch * stride_batch + head * stride_head +
          address.raw_token * stride_token + element;
      payload = load_fp16_vector(input, input_offset);
    }
    const int64_t virtual_token =
        logical_patch * kRasterPatchTokens + local_token;
    const int64_t output_offset =
        ((batch * num_heads + head) * virtual_tokens + virtual_token) *
            HeadDim +
        element;
    *reinterpret_cast<uint4*>(output + output_offset) = payload;
  }
}

template <int32_t HeadDim, typename InputT>
__global__ void pack_raster_kv_to_fp16_kernel(
    const InputT* __restrict__ key,
    const InputT* __restrict__ value,
    half* __restrict__ packed_key,
    half* __restrict__ packed_value,
    Raster8x16Layout layout,
    int64_t num_heads,
    int64_t raw_tokens,
    int64_t virtual_tokens,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_stride_token) {
  constexpr int32_t vectors_per_token =
      HeadDim / kHalfElementsPerVector;
  constexpr int32_t vectors_per_patch =
      kRasterPatchTokens * vectors_per_token;

  const int64_t logical_patch = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  for (int32_t vector = threadIdx.x; vector < vectors_per_patch;
       vector += blockDim.x) {
    const int32_t local_token = vector / vectors_per_token;
    const int32_t vector_in_token = vector % vectors_per_token;
    const int32_t element = vector_in_token * kHalfElementsPerVector;
    const RasterTokenAddress address =
        layout.logical_token(logical_patch, local_token);

    uint4 key_payload = make_uint4(0, 0, 0, 0);
    uint4 value_payload = make_uint4(0, 0, 0, 0);
    if (address.valid) {
      const int64_t key_offset = batch * key_stride_batch +
          head * key_stride_head + address.raw_token * key_stride_token +
          element;
      const int64_t value_offset = batch * value_stride_batch +
          head * value_stride_head + address.raw_token * value_stride_token +
          element;
      key_payload = load_fp16_vector(key, key_offset);
      value_payload = load_fp16_vector(value, value_offset);
    }
    const int64_t virtual_token =
        logical_patch * kRasterPatchTokens + local_token;
    const int64_t output_offset =
        ((batch * num_heads + head) * virtual_tokens + virtual_token) *
            HeadDim +
        element;
    *reinterpret_cast<uint4*>(packed_key + output_offset) = key_payload;
    *reinterpret_cast<uint4*>(packed_value + output_offset) = value_payload;
  }
}

template <int32_t HeadDim, typename InputT>
__global__ void pack_indexed_k64_qkv_to_fp16_kernel(
    const InputT* __restrict__ query,
    const InputT* __restrict__ key,
    const InputT* __restrict__ value,
    const int64_t* __restrict__ token_indices,
    const bool* __restrict__ slot_valid,
    half* __restrict__ packed_query,
    half* __restrict__ packed_key,
    half* __restrict__ packed_value,
    int64_t num_heads,
    int64_t physical_tokens,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t query_stride_token,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_stride_token) {
  constexpr int32_t vectors_per_token =
      HeadDim / kHalfElementsPerVector;
  constexpr int32_t vectors_per_block = 64 * vectors_per_token;
  const int64_t physical_block = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;

  for (int32_t vector = threadIdx.x; vector < vectors_per_block;
       vector += blockDim.x) {
    const int32_t local_token = vector / vectors_per_token;
    const int32_t vector_in_token = vector % vectors_per_token;
    const int32_t element = vector_in_token * kHalfElementsPerVector;
    const int64_t physical_token = physical_block * 64 + local_token;
    uint4 query_payload = make_uint4(0, 0, 0, 0);
    uint4 key_payload = make_uint4(0, 0, 0, 0);
    uint4 value_payload = make_uint4(0, 0, 0, 0);
    if (slot_valid[physical_token]) {
      const int64_t raw_token = token_indices[physical_token];
      const int64_t query_offset = batch * query_stride_batch +
          head * query_stride_head + raw_token * query_stride_token + element;
      const int64_t key_offset = batch * key_stride_batch +
          head * key_stride_head + raw_token * key_stride_token + element;
      const int64_t value_offset = batch * value_stride_batch +
          head * value_stride_head + raw_token * value_stride_token + element;
      query_payload = load_fp16_vector(query, query_offset);
      key_payload = load_fp16_vector(key, key_offset);
      value_payload = load_fp16_vector(value, value_offset);
    }
    const int64_t output_offset =
        ((batch * num_heads + head) * physical_tokens + physical_token) *
            HeadDim +
        element;
    *reinterpret_cast<uint4*>(packed_query + output_offset) = query_payload;
    *reinterpret_cast<uint4*>(packed_key + output_offset) = key_payload;
    *reinterpret_cast<uint4*>(packed_value + output_offset) = value_payload;
  }
}

template <int32_t HeadDim, typename InputT>
__global__ void pack_h3_k64_qkv_to_fp16_kernel(
    const InputT* __restrict__ query,
    const InputT* __restrict__ key,
    const InputT* __restrict__ value,
    const int64_t* __restrict__ video_token_indices,
    const bool* __restrict__ video_slot_valid,
    half* __restrict__ packed_video_query,
    half* __restrict__ packed_key,
    half* __restrict__ packed_value,
    int64_t num_heads,
    int64_t video_physical_tokens,
    int64_t prefix_tokens,
    int64_t prefix_blocks,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t query_stride_token,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_stride_token) {
  constexpr int32_t vectors_per_token =
      HeadDim / kHalfElementsPerVector;
  constexpr int32_t vectors_per_block = 64 * vectors_per_token;
  const int64_t output_block = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  const bool is_prefix = output_block < prefix_blocks;

  for (int32_t vector = threadIdx.x; vector < vectors_per_block;
       vector += blockDim.x) {
    const int32_t local_token = vector / vectors_per_token;
    const int32_t vector_in_token = vector % vectors_per_token;
    const int32_t element = vector_in_token * kHalfElementsPerVector;
    uint4 query_payload = make_uint4(0, 0, 0, 0);
    uint4 key_payload = make_uint4(0, 0, 0, 0);
    uint4 value_payload = make_uint4(0, 0, 0, 0);
    int64_t video_physical_token = 0;

    if (is_prefix) {
      const int64_t raw_token = output_block * 64 + local_token;
      if (raw_token < prefix_tokens) {
        const int64_t key_offset = batch * key_stride_batch +
            head * key_stride_head + raw_token * key_stride_token + element;
        const int64_t value_offset = batch * value_stride_batch +
            head * value_stride_head + raw_token * value_stride_token + element;
        key_payload = load_fp16_vector(key, key_offset);
        value_payload = load_fp16_vector(value, value_offset);
      }
    } else {
      video_physical_token =
          (output_block - prefix_blocks) * 64 + local_token;
      if (video_slot_valid[video_physical_token]) {
        const int64_t raw_token =
            prefix_tokens + video_token_indices[video_physical_token];
        const int64_t query_offset = batch * query_stride_batch +
            head * query_stride_head + raw_token * query_stride_token + element;
        const int64_t key_offset = batch * key_stride_batch +
            head * key_stride_head + raw_token * key_stride_token + element;
        const int64_t value_offset = batch * value_stride_batch +
            head * value_stride_head + raw_token * value_stride_token + element;
        query_payload = load_fp16_vector(query, query_offset);
        key_payload = load_fp16_vector(key, key_offset);
        value_payload = load_fp16_vector(value, value_offset);
      }
      const int64_t query_output_offset =
          ((batch * num_heads + head) * video_physical_tokens +
           video_physical_token) * HeadDim + element;
      *reinterpret_cast<uint4*>(packed_video_query + query_output_offset) =
          query_payload;
    }
    const int64_t key_output_token = output_block * 64 + local_token;
    const int64_t key_output_offset =
        ((batch * num_heads + head) *
             (prefix_blocks * 64 + video_physical_tokens) +
         key_output_token) * HeadDim + element;
    *reinterpret_cast<uint4*>(packed_key + key_output_offset) = key_payload;
    *reinterpret_cast<uint4*>(packed_value + key_output_offset) = value_payload;
  }
}

template <int32_t HeadDim, typename InputT>
void launch_raster_pack(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& packed_query,
    torch::Tensor& packed_key,
    torch::Tensor& packed_value,
    Raster8x16Layout layout,
    cudaStream_t stream) {
  const int64_t batch_size = query.size(0);
  const int64_t query_heads = query.size(1);
  const int64_t kv_heads = key.size(1);
  const int64_t raw_tokens = query.size(2);
  const int64_t virtual_tokens = packed_query.size(2);

  const dim3 query_grid(
      static_cast<unsigned int>(layout.patch_count),
      static_cast<unsigned int>(query_heads),
      static_cast<unsigned int>(batch_size));
  pack_raster_operand_to_fp16_kernel<HeadDim, InputT><<<
      query_grid, kPackThreads, 0, stream>>>(
      reinterpret_cast<const InputT*>(query.data_ptr()),
      reinterpret_cast<half*>(packed_query.data_ptr<at::Half>()),
      layout,
      query_heads,
      raw_tokens,
      virtual_tokens,
      query.stride(0), query.stride(1), query.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 kv_grid(
      static_cast<unsigned int>(layout.patch_count),
      static_cast<unsigned int>(kv_heads),
      static_cast<unsigned int>(batch_size));
  pack_raster_kv_to_fp16_kernel<HeadDim, InputT><<<
      kv_grid, kPackThreads, 0, stream>>>(
      reinterpret_cast<const InputT*>(key.data_ptr()),
      reinterpret_cast<const InputT*>(value.data_ptr()),
      reinterpret_cast<half*>(packed_key.data_ptr<at::Half>()),
      reinterpret_cast<half*>(packed_value.data_ptr<at::Half>()),
      layout,
      kv_heads,
      raw_tokens,
      virtual_tokens,
      key.stride(0), key.stride(1), key.stride(2),
      value.stride(0), value.stride(1), value.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace
}  // namespace mpa::attention

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
pack_indexed_k64_qkv_fp16(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor token_indices,
    torch::Tensor slot_valid) {
  using namespace mpa::attention;
  check_raw_hnd(query, "query");
  check_raw_hnd(key, "key");
  check_raw_hnd(value, "value");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device(),
      "query, key, and value must share one CUDA device");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value.scalar_type(),
      "query, key, and value dtypes must match");
  TORCH_CHECK(
      query.sizes() == key.sizes() && query.sizes() == value.sizes(),
      "indexed K64 query, key, and value shapes must match");
  TORCH_CHECK(
      token_indices.is_cuda() && token_indices.device() == query.device() &&
          token_indices.scalar_type() == at::ScalarType::Long &&
          token_indices.dim() == 1 && token_indices.is_contiguous(),
      "token_indices must be contiguous CUDA int64 [physical_tokens]");
  TORCH_CHECK(
      slot_valid.is_cuda() && slot_valid.device() == query.device() &&
          slot_valid.scalar_type() == at::ScalarType::Bool &&
          slot_valid.dim() == 1 && slot_valid.is_contiguous(),
      "slot_valid must be contiguous CUDA bool [physical_tokens]");
  const int64_t physical_tokens = token_indices.numel();
  TORCH_CHECK(
      physical_tokens > 0 && physical_tokens % 64 == 0 &&
          slot_valid.numel() == physical_tokens,
      "indexed K64 metadata must have matching positive K64 capacity");

  c10::cuda::CUDAGuard device_guard(query.device());
  check_raster_and_grid(
      query, physical_tokens / 64, query.size(1));
  const auto fp16_options = query.options().dtype(at::ScalarType::Half);
  auto packed_query = torch::empty(
      {query.size(0), query.size(1), physical_tokens, query.size(3)},
      fp16_options);
  auto packed_key = torch::empty_like(packed_query);
  auto packed_value = torch::empty_like(packed_query);
  const dim3 grid(
      static_cast<unsigned int>(physical_tokens / 64),
      static_cast<unsigned int>(query.size(1)),
      static_cast<unsigned int>(query.size(0)));
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(query.get_device());
  const auto launch = [&](auto head_dim_tag, auto* input_tag) {
    constexpr int32_t HeadDim = decltype(head_dim_tag)::value;
    using InputT = std::remove_pointer_t<decltype(input_tag)>;
    pack_indexed_k64_qkv_to_fp16_kernel<HeadDim, InputT><<<
        grid, kPackThreads, 0, stream>>>(
        reinterpret_cast<const InputT*>(query.data_ptr()),
        reinterpret_cast<const InputT*>(key.data_ptr()),
        reinterpret_cast<const InputT*>(value.data_ptr()),
        token_indices.data_ptr<int64_t>(),
        slot_valid.data_ptr<bool>(),
        reinterpret_cast<half*>(packed_query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(packed_key.data_ptr<at::Half>()),
        reinterpret_cast<half*>(packed_value.data_ptr<at::Half>()),
        query.size(1),
        physical_tokens,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2));
  };
  if (query.scalar_type() == at::ScalarType::Half) {
    if (query.size(3) == 64) {
      launch(std::integral_constant<int32_t, 64>{},
             static_cast<half*>(nullptr));
    } else {
      launch(std::integral_constant<int32_t, 128>{},
             static_cast<half*>(nullptr));
    }
  } else if (query.size(3) == 64) {
    launch(std::integral_constant<int32_t, 64>{},
           static_cast<nv_bfloat16*>(nullptr));
  } else {
    launch(std::integral_constant<int32_t, 128>{},
           static_cast<nv_bfloat16*>(nullptr));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_query, packed_key, packed_value};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
pack_h3_k64_qkv_fp16(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor video_token_indices,
    torch::Tensor video_slot_valid,
    int64_t prefix_tokens) {
  using namespace mpa::attention;
  check_raw_hnd(query, "query");
  check_raw_hnd(key, "key");
  check_raw_hnd(value, "value");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device(),
      "query, key, and value must share one CUDA device");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value.scalar_type(),
      "query, key, and value dtypes must match");
  TORCH_CHECK(
      query.sizes() == key.sizes() && query.sizes() == value.sizes(),
      "H3 K64 query, key, and value shapes must match");
  TORCH_CHECK(
      prefix_tokens >= 0 && prefix_tokens < query.size(2),
      "prefix_tokens must be nonnegative and smaller than the sequence");
  TORCH_CHECK(
      video_token_indices.is_cuda() &&
          video_token_indices.device() == query.device() &&
          video_token_indices.scalar_type() == at::ScalarType::Long &&
          video_token_indices.dim() == 1 && video_token_indices.is_contiguous(),
      "video_token_indices must be contiguous CUDA int64 [physical_tokens]");
  TORCH_CHECK(
      video_slot_valid.is_cuda() &&
          video_slot_valid.device() == query.device() &&
          video_slot_valid.scalar_type() == at::ScalarType::Bool &&
          video_slot_valid.dim() == 1 && video_slot_valid.is_contiguous(),
      "video_slot_valid must be contiguous CUDA bool [physical_tokens]");
  const int64_t video_physical_tokens = video_token_indices.numel();
  TORCH_CHECK(
      video_physical_tokens > 0 && video_physical_tokens % 64 == 0 &&
          video_slot_valid.numel() == video_physical_tokens,
      "H3 indexed video metadata must have matching positive K64 capacity");
  const int64_t prefix_blocks = (prefix_tokens + 63) / 64;
  const int64_t key_physical_tokens =
      prefix_blocks * 64 + video_physical_tokens;

  c10::cuda::CUDAGuard device_guard(query.device());
  check_raster_and_grid(
      query, prefix_blocks + video_physical_tokens / 64, query.size(1));
  const auto fp16_options = query.options().dtype(at::ScalarType::Half);
  auto packed_query = torch::empty(
      {query.size(0), query.size(1), video_physical_tokens, query.size(3)},
      fp16_options);
  auto packed_key = torch::empty(
      {query.size(0), query.size(1), key_physical_tokens, query.size(3)},
      fp16_options);
  auto packed_value = torch::empty_like(packed_key);
  const dim3 grid(
      static_cast<unsigned int>(prefix_blocks + video_physical_tokens / 64),
      static_cast<unsigned int>(query.size(1)),
      static_cast<unsigned int>(query.size(0)));
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(query.get_device());
  const auto launch = [&](auto head_dim_tag, auto* input_tag) {
    constexpr int32_t HeadDim = decltype(head_dim_tag)::value;
    using InputT = std::remove_pointer_t<decltype(input_tag)>;
    pack_h3_k64_qkv_to_fp16_kernel<HeadDim, InputT><<<
        grid, kPackThreads, 0, stream>>>(
        reinterpret_cast<const InputT*>(query.data_ptr()),
        reinterpret_cast<const InputT*>(key.data_ptr()),
        reinterpret_cast<const InputT*>(value.data_ptr()),
        video_token_indices.data_ptr<int64_t>(),
        video_slot_valid.data_ptr<bool>(),
        reinterpret_cast<half*>(packed_query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(packed_key.data_ptr<at::Half>()),
        reinterpret_cast<half*>(packed_value.data_ptr<at::Half>()),
        query.size(1), video_physical_tokens, prefix_tokens, prefix_blocks,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2));
  };
  if (query.scalar_type() == at::ScalarType::Half) {
    if (query.size(3) == 64) {
      launch(std::integral_constant<int32_t, 64>{},
             static_cast<half*>(nullptr));
    } else {
      launch(std::integral_constant<int32_t, 128>{},
             static_cast<half*>(nullptr));
    }
  } else if (query.size(3) == 64) {
    launch(std::integral_constant<int32_t, 64>{},
           static_cast<nv_bfloat16*>(nullptr));
  } else {
    launch(std::integral_constant<int32_t, 128>{},
           static_cast<nv_bfloat16*>(nullptr));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_query, packed_key, packed_value};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
pack_raster_qkv_fp16(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t frames,
    int64_t height,
    int64_t width) {
  using namespace mpa::attention;
  check_raw_hnd(query, "query");
  check_raw_hnd(key, "key");
  check_raw_hnd(value, "value");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device(),
      "query, key, and value must share one CUDA device");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value.scalar_type(),
      "query, key, and value dtypes must match");
  TORCH_CHECK(
      query.size(0) == key.size(0) && query.size(0) == value.size(0),
      "query, key, and value batch dimensions must match");
  TORCH_CHECK(
      key.size(1) == value.size(1),
      "key and value head dimensions must match");
  TORCH_CHECK(
      query.size(1) % key.size(1) == 0,
      "query head count must be divisible by KV head count");
  TORCH_CHECK(
      query.size(2) == key.size(2) && query.size(2) == value.size(2),
      "query, key, and value token dimensions must match");
  TORCH_CHECK(
      query.size(3) == key.size(3) && query.size(3) == value.size(3),
      "query, key, and value head dimensions D must match");

  const Raster8x16Layout layout = checked_layout(frames, height, width);
  const int64_t expected_tokens = checked_positive_product(
      frames,
      checked_positive_product(height, width, "Y*X"),
      "F*Y*X");
  TORCH_CHECK(
      query.size(2) == expected_tokens,
      "input token dimension must equal frames*height*width");
  const int64_t virtual_tokens = checked_positive_product(
      layout.patch_count, kRasterPatchTokens, "R*128 virtual tokens");

  c10::cuda::CUDAGuard device_guard(query.device());
  check_raster_and_grid(
      query, layout.patch_count, std::max(query.size(1), key.size(1)));
  const auto fp16_options = query.options().dtype(at::ScalarType::Half);
  auto packed_query = torch::empty(
      {query.size(0), query.size(1), virtual_tokens, query.size(3)},
      fp16_options);
  auto packed_key = torch::empty(
      {key.size(0), key.size(1), virtual_tokens, key.size(3)},
      fp16_options);
  auto packed_value = torch::empty_like(packed_key);

  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(query.get_device());
  if (query.scalar_type() == at::ScalarType::Half) {
    if (query.size(3) == 64) {
      launch_raster_pack<64, half>(
          query,
          key,
          value,
          packed_query,
          packed_key,
          packed_value,
          layout,
          stream);
    } else {
      launch_raster_pack<128, half>(
          query,
          key,
          value,
          packed_query,
          packed_key,
          packed_value,
          layout,
          stream);
    }
  } else if (query.size(3) == 64) {
    launch_raster_pack<64, nv_bfloat16>(
        query,
        key,
        value,
        packed_query,
        packed_key,
        packed_value,
        layout,
        stream);
  } else {
    launch_raster_pack<128, nv_bfloat16>(
        query,
        key,
        value,
        packed_query,
        packed_key,
        packed_value,
        layout,
        stream);
  }
  return {packed_query, packed_key, packed_value};
}

torch::Tensor pack_raster_v_fp16(
    torch::Tensor value,
    int64_t frames,
    int64_t height,
    int64_t width) {
  using namespace mpa::attention;
  check_raw_hnd(value, "value");
  const Raster8x16Layout layout = checked_layout(frames, height, width);
  const int64_t expected_tokens = checked_positive_product(
      frames,
      checked_positive_product(height, width, "Y*X"),
      "F*Y*X");
  TORCH_CHECK(
      value.size(2) == expected_tokens,
      "value token dimension must equal frames*height*width");
  const int64_t virtual_tokens = checked_positive_product(
      layout.patch_count, kRasterPatchTokens, "R*128 virtual tokens");

  c10::cuda::CUDAGuard device_guard(value.device());
  check_raster_and_grid(value, layout.patch_count, value.size(1));
  auto packed_value = torch::empty(
      {value.size(0), value.size(1), virtual_tokens, value.size(3)},
      value.options().dtype(at::ScalarType::Half));
  const dim3 grid(
      static_cast<unsigned int>(layout.patch_count),
      static_cast<unsigned int>(value.size(1)),
      static_cast<unsigned int>(value.size(0)));
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(value.get_device());

  const auto launch = [&](auto head_dim_tag, auto* input_tag) {
    constexpr int32_t HeadDim = decltype(head_dim_tag)::value;
    using InputT = std::remove_pointer_t<decltype(input_tag)>;
    pack_raster_operand_to_fp16_kernel<HeadDim, InputT><<<
        grid, kPackThreads, 0, stream>>>(
        reinterpret_cast<const InputT*>(value.data_ptr()),
        reinterpret_cast<half*>(packed_value.data_ptr<at::Half>()),
        layout,
        value.size(1),
        value.size(2),
        virtual_tokens,
        value.stride(0), value.stride(1), value.stride(2));
  };
  if (value.scalar_type() == at::ScalarType::Half) {
    if (value.size(3) == 64) {
      launch(std::integral_constant<int32_t, 64>{},
             static_cast<half*>(nullptr));
    } else {
      launch(std::integral_constant<int32_t, 128>{},
             static_cast<half*>(nullptr));
    }
  } else if (value.size(3) == 64) {
    launch(std::integral_constant<int32_t, 64>{},
           static_cast<nv_bfloat16*>(nullptr));
  } else {
    launch(std::integral_constant<int32_t, 128>{},
           static_cast<nv_bfloat16*>(nullptr));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed_value;
}

std::tuple<torch::Tensor, torch::Tensor> pack_raster_kv_fp16(
    torch::Tensor key,
    torch::Tensor value,
    int64_t frames,
    int64_t height,
    int64_t width) {
  using namespace mpa::attention;
  check_raw_hnd(key, "key");
  check_raw_hnd(value, "value");
  TORCH_CHECK(
      key.device() == value.device(),
      "key and value must share one CUDA device");
  TORCH_CHECK(
      key.scalar_type() == value.scalar_type(),
      "key and value dtypes must match");
  TORCH_CHECK(
      key.sizes() == value.sizes(),
      "key and value shapes must match");

  const Raster8x16Layout layout = checked_layout(frames, height, width);
  const int64_t expected_tokens = checked_positive_product(
      frames,
      checked_positive_product(height, width, "Y*X"),
      "F*Y*X");
  TORCH_CHECK(
      key.size(2) == expected_tokens,
      "key/value token dimension must equal frames*height*width");
  const int64_t virtual_tokens = checked_positive_product(
      layout.patch_count, kRasterPatchTokens, "R*128 virtual tokens");

  c10::cuda::CUDAGuard device_guard(key.device());
  check_raster_and_grid(key, layout.patch_count, key.size(1));
  const auto fp16_options = key.options().dtype(at::ScalarType::Half);
  auto packed_key = torch::empty(
      {key.size(0), key.size(1), virtual_tokens, key.size(3)},
      fp16_options);
  auto packed_value = torch::empty_like(packed_key);
  const dim3 grid(
      static_cast<unsigned int>(layout.patch_count),
      static_cast<unsigned int>(key.size(1)),
      static_cast<unsigned int>(key.size(0)));
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(key.get_device());

  const auto launch = [&](auto head_dim_tag, auto* input_tag) {
    constexpr int32_t HeadDim = decltype(head_dim_tag)::value;
    using InputT = std::remove_pointer_t<decltype(input_tag)>;
    pack_raster_kv_to_fp16_kernel<HeadDim, InputT><<<
        grid, kPackThreads, 0, stream>>>(
        reinterpret_cast<const InputT*>(key.data_ptr()),
        reinterpret_cast<const InputT*>(value.data_ptr()),
        reinterpret_cast<half*>(packed_key.data_ptr<at::Half>()),
        reinterpret_cast<half*>(packed_value.data_ptr<at::Half>()),
        layout,
        key.size(1),
        key.size(2),
        virtual_tokens,
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2));
  };
  if (key.scalar_type() == at::ScalarType::Half) {
    if (key.size(3) == 64) {
      launch(std::integral_constant<int32_t, 64>{},
             static_cast<half*>(nullptr));
    } else {
      launch(std::integral_constant<int32_t, 128>{},
             static_cast<half*>(nullptr));
    }
  } else if (key.size(3) == 64) {
    launch(std::integral_constant<int32_t, 64>{},
           static_cast<nv_bfloat16*>(nullptr));
  } else {
    launch(std::integral_constant<int32_t, 128>{},
           static_cast<nv_bfloat16*>(nullptr));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_key, packed_value};
}
