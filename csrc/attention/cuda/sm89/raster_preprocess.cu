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
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include "api.h"
#include "primitives/numeric_conversion.cuh"
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

template <typename InputT>
__device__ __forceinline__ half load_narrowed_scalar(
    const InputT* input,
    int64_t offset);

template <>
__device__ __forceinline__ half load_narrowed_scalar<half>(
    const half* input,
    int64_t offset) {
  return input[offset];
}

template <>
__device__ __forceinline__ half load_narrowed_scalar<nv_bfloat16>(
    const nv_bfloat16* input,
    int64_t offset) {
  return __float2half_rn(__bfloat162float(input[offset]));
}

__device__ __forceinline__ float warp_allreduce_max(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, mask));
  }
  return value;
}

template <
    typename InputT,
    int32_t QueryBlock,
    bool SmoothK,
    bool HasMaxPool>
__global__ void prepare_h3_sm89_qk_single_load_kernel(
    const InputT* __restrict__ query,
    const InputT* __restrict__ key,
    const int64_t* __restrict__ video_token_indices,
    const bool* __restrict__ video_slot_valid,
    const int32_t* __restrict__ video_valid_counts,
    half* __restrict__ q_pool,
    half* __restrict__ k_pool,
    half* __restrict__ q_max_pool,
    half* __restrict__ k_max_pool,
    half* __restrict__ packed_q,
    half* __restrict__ packed_k,
    int8_t* __restrict__ q8,
    int8_t* __restrict__ k8,
    float* __restrict__ q_scale,
    float* __restrict__ k_scale,
    float* __restrict__ k_stage_sum,
    int64_t heads,
    int64_t video_blocks,
    int64_t prefix_tokens,
    int64_t prefix_blocks,
    int64_t video_physical_tokens,
    int64_t key_physical_tokens,
    int64_t q_stride_batch,
    int64_t q_stride_head,
    int64_t q_stride_token,
    int64_t k_stride_batch,
    int64_t k_stride_head,
    int64_t k_stride_token) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  constexpr int32_t HeadDim = 128;
  constexpr int32_t Warps = HeadDim / 32;
  __shared__ half tile[QueryBlock * HeadDim];
  __shared__ int64_t staged_token_indices[QueryBlock];
  __shared__ bool staged_slot_valid[QueryBlock];
  __shared__ float warp_amax[2][Warps];
  __shared__ float block_scale[2];

  const int64_t task = blockIdx.x;
  const int64_t batch = blockIdx.z;
  const int64_t head = blockIdx.y;
  const int64_t head_batch = batch * heads + head;
  const bool is_video_query = task < video_blocks;
  const bool is_video_key =
      task >= video_blocks && task < 2 * video_blocks;
  const bool is_prefix_key = task >= 2 * video_blocks;
  const bool is_query = is_video_query;
  const int64_t logical_block = is_video_query
      ? task
      : (is_video_key ? task - video_blocks : -1);
  const int64_t prefix_block =
      is_prefix_key ? task - 2 * video_blocks : -1;
  const int32_t task_tokens = is_prefix_key ? 64 : QueryBlock;
  const int32_t channel = threadIdx.x;
  const int32_t lane = channel & 31;
  const int32_t warp = channel >> 5;

  if (!is_prefix_key && channel < task_tokens) {
    staged_token_indices[channel] =
        video_token_indices[logical_block * QueryBlock + channel];
    staged_slot_valid[channel] =
        video_slot_valid[logical_block * QueryBlock + channel];
  }
  __syncthreads();

  const InputT* input = is_query ? query : key;
  const int64_t stride_batch = is_query ? q_stride_batch : k_stride_batch;
  const int64_t stride_head = is_query ? q_stride_head : k_stride_head;
  const int64_t stride_token = is_query ? q_stride_token : k_stride_token;
  const int64_t output_tokens =
      is_query ? video_physical_tokens : key_physical_tokens;
  half* packed = is_query ? packed_q : packed_k;
  int8_t* quantized_output = is_query ? q8 : k8;
  float local_amax[2] = {0.0f, 0.0f};
  float pool_sum = 0.0f;
  float pool_max = -CUDART_INF_F;
  float stage_sum[2] = {0.0f, 0.0f};

#pragma unroll 1
  for (int32_t token = 0; token < task_tokens; ++token) {
    const bool valid = is_prefix_key
        ? prefix_block * 64 + token < prefix_tokens
        : staged_slot_valid[token];
    const int64_t raw_token = is_prefix_key
        ? prefix_block * 64 + token
        : prefix_tokens + staged_token_indices[token];
    half narrowed = __float2half_rn(0.0f);
    if (valid) {
      narrowed = load_narrowed_scalar(
          input,
          batch * stride_batch + head * stride_head +
              raw_token * stride_token + channel);
    }
    tile[token * HeadDim + channel] = narrowed;
    const int64_t natural_row = is_query
        ? logical_block * QueryBlock + token
        : (is_prefix_key
              ? prefix_block * 64 + token
              : prefix_blocks * 64 + logical_block * QueryBlock + token);
    packed[(head_batch * output_tokens + natural_row) * HeadDim + channel] =
        narrowed;

    const float value = __half2float(narrowed);
    if (!is_prefix_key && valid) {
      pool_sum += value;
      if constexpr (HasMaxPool) {
        pool_max = fmaxf(pool_max, value);
      }
    }
    const int32_t group =
        !is_query && !is_prefix_key && QueryBlock == 128 ? token / 64 : 0;
    if constexpr (SmoothK) {
      if (!is_query && valid) stage_sum[group] += value;
    }
    if constexpr (!SmoothK) {
      local_amax[group] = fmaxf(local_amax[group], fabsf(value));
    } else if (is_query) {
      local_amax[0] = fmaxf(local_amax[0], fabsf(value));
    }
  }

  if (!is_prefix_key) {
    const int32_t valid_count = video_valid_counts[logical_block];
    const int64_t pool_offset =
        (head_batch * video_blocks + logical_block) * HeadDim + channel;
    half* pool = is_query ? q_pool : k_pool;
    pool[pool_offset] =
        __float2half_rn(pool_sum / static_cast<float>(valid_count));
    if constexpr (HasMaxPool) {
      half* max_pool = is_query ? q_max_pool : k_max_pool;
      max_pool[pool_offset] = __float2half_rn(pool_max);
    }
  }

  if constexpr (SmoothK) {
    if (!is_query) {
      const int32_t groups =
          !is_prefix_key && QueryBlock == 128 ? 2 : 1;
#pragma unroll
      for (int32_t group = 0; group < 2; ++group) {
        if (group < groups) {
          const int64_t physical_stage = is_prefix_key
              ? prefix_block
              : prefix_blocks + logical_block * (QueryBlock / 64) + group;
          k_stage_sum[
              (head_batch * (key_physical_tokens / 64) + physical_stage) *
                  HeadDim +
              channel] = stage_sum[group];
        }
      }
    }
  }

  const bool quantize_here = !SmoothK || is_query;
  if (!quantize_here) return;
  const int32_t groups =
      !is_query && !is_prefix_key && QueryBlock == 128 ? 2 : 1;
#pragma unroll
  for (int32_t group = 0; group < 2; ++group) {
    if (group < groups) {
      const float reduced = warp_allreduce_max(local_amax[group]);
      if (lane == 0) warp_amax[group][warp] = reduced;
    }
  }
  __syncthreads();
  if (channel < groups) {
    float amax = warp_amax[channel][0];
#pragma unroll
    for (int32_t warp_index = 1; warp_index < Warps; ++warp_index) {
      amax = fmaxf(amax, warp_amax[channel][warp_index]);
    }
    const float scale = amax / 127.0f + 1.0e-7f;
    block_scale[channel] = scale;
    if (is_query) {
      q_scale[(head_batch * video_blocks) + logical_block] = scale;
    } else {
      const int64_t physical_stage = is_prefix_key
          ? prefix_block
          : prefix_blocks + logical_block * (QueryBlock / 64) + channel;
      k_scale[head_batch * (key_physical_tokens / 64) + physical_stage] = scale;
    }
  }
  __syncthreads();

#pragma unroll 1
  for (int32_t token = 0; token < task_tokens; ++token) {
    const int32_t group =
        !is_query && !is_prefix_key && QueryBlock == 128 ? token / 64 : 0;
    float quantized =
        __half2float(tile[token * HeadDim + channel]) / block_scale[group];
    quantized += quantized >= 0.0f ? 0.5f : -0.5f;
    const int64_t natural_row = is_query
        ? logical_block * QueryBlock + token
        : (is_prefix_key
              ? prefix_block * 64 + token
              : prefix_blocks * 64 + logical_block * QueryBlock + token);
    quantized_output[
        (head_batch * output_tokens + natural_row) * HeadDim + channel] =
        static_cast<int8_t>(__float2int_rz(quantized));
  }
}

// Q128 queries need one 128-row reduction, but each K scale still owns one
// K64 stage.  Keeping Q and K in the same Q128 specialization would therefore
// charge every video-K and prefix-K CTA for the query kernel's 32 KiB tile.
// Stream the two K64 halves through one 16 KiB tile instead: every raw K value
// is still fetched exactly once, while the K-side occupancy stays identical to
// the native K64 preparation path.
template <typename InputT, bool SmoothK, bool HasMaxPool>
__global__ void prepare_h3_sm89_q128_key_single_load_kernel(
    const InputT* __restrict__ key,
    const int64_t* __restrict__ video_token_indices,
    const bool* __restrict__ video_slot_valid,
    const int32_t* __restrict__ video_valid_counts,
    half* __restrict__ k_pool,
    half* __restrict__ k_max_pool,
    half* __restrict__ packed_k,
    int8_t* __restrict__ k8,
    float* __restrict__ k_scale,
    float* __restrict__ k_stage_sum,
    int64_t heads,
    int64_t video_blocks,
    int64_t prefix_tokens,
    int64_t prefix_blocks,
    int64_t key_physical_tokens,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_token) {
  constexpr int32_t HeadDim = 128;
  constexpr int32_t StageTokens = 64;
  constexpr int32_t Warps = HeadDim / 32;
  __shared__ half tile[SmoothK ? 1 : StageTokens * HeadDim];
  __shared__ int64_t staged_token_indices[StageTokens];
  __shared__ bool staged_slot_valid[StageTokens];
  __shared__ float warp_amax[Warps];
  __shared__ float block_scale;

  const int64_t task = blockIdx.x;
  const int64_t batch = blockIdx.z;
  const int64_t head = blockIdx.y;
  const int64_t head_batch = batch * heads + head;
  const bool is_video = task < video_blocks;
  const int64_t logical_block = is_video ? task : -1;
  const int64_t prefix_block = is_video ? -1 : task - video_blocks;
  const int32_t groups = is_video ? 2 : 1;
  const int32_t channel = threadIdx.x;
  const int32_t lane = channel & 31;
  const int32_t warp = channel >> 5;
  float pool_sum = 0.0f;
  float pool_max = -CUDART_INF_F;

#pragma unroll
  for (int32_t group = 0; group < 2; ++group) {
    if (group >= groups) break;
    if (is_video && channel < StageTokens) {
      const int64_t slot =
          logical_block * 128 + group * StageTokens + channel;
      staged_token_indices[channel] = video_token_indices[slot];
      staged_slot_valid[channel] = video_slot_valid[slot];
    }
    __syncthreads();

    float local_amax = 0.0f;
    float stage_sum = 0.0f;
#pragma unroll 1
    for (int32_t token = 0; token < StageTokens; ++token) {
      const bool valid = is_video
          ? staged_slot_valid[token]
          : prefix_block * StageTokens + token < prefix_tokens;
      const int64_t raw_token = is_video
          ? prefix_tokens + staged_token_indices[token]
          : prefix_block * StageTokens + token;
      half narrowed = __float2half_rn(0.0f);
      if (valid) {
        narrowed = load_narrowed_scalar(
            key,
            batch * stride_batch + head * stride_head +
                raw_token * stride_token + channel);
      }
      const int64_t physical_stage = is_video
          ? prefix_blocks + logical_block * 2 + group
          : prefix_block;
      const int64_t natural_row = physical_stage * StageTokens + token;
      packed_k[
          (head_batch * key_physical_tokens + natural_row) * HeadDim +
          channel] = narrowed;
      const float value = __half2float(narrowed);
      if (is_video && valid) {
        pool_sum += value;
        if constexpr (HasMaxPool) {
          pool_max = fmaxf(pool_max, value);
        }
      }
      if constexpr (SmoothK) {
        if (valid) stage_sum += value;
      } else {
        tile[token * HeadDim + channel] = narrowed;
        local_amax = fmaxf(local_amax, fabsf(value));
      }
    }

    const int64_t physical_stage = is_video
        ? prefix_blocks + logical_block * 2 + group
        : prefix_block;
    if constexpr (SmoothK) {
      k_stage_sum[
          (head_batch * (key_physical_tokens / StageTokens) + physical_stage) *
              HeadDim +
          channel] = stage_sum;
    } else {
      const float reduced = warp_allreduce_max(local_amax);
      if (lane == 0) warp_amax[warp] = reduced;
      __syncthreads();
      if (channel == 0) {
        float amax = warp_amax[0];
#pragma unroll
        for (int32_t warp_index = 1; warp_index < Warps; ++warp_index) {
          amax = fmaxf(amax, warp_amax[warp_index]);
        }
        block_scale = amax / 127.0f + 1.0e-7f;
        k_scale[head_batch * (key_physical_tokens / StageTokens) +
                physical_stage] = block_scale;
      }
      __syncthreads();
#pragma unroll 1
      for (int32_t token = 0; token < StageTokens; ++token) {
        float quantized =
            __half2float(tile[token * HeadDim + channel]) / block_scale;
        quantized += quantized >= 0.0f ? 0.5f : -0.5f;
        k8[(head_batch * key_physical_tokens +
            physical_stage * StageTokens + token) * HeadDim + channel] =
            static_cast<int8_t>(__float2int_rz(quantized));
      }
    }
    __syncthreads();
  }

  if (is_video) {
    const int64_t pool_offset =
        (head_batch * video_blocks + logical_block) * HeadDim + channel;
    k_pool[pool_offset] = __float2half_rn(
        pool_sum / static_cast<float>(video_valid_counts[logical_block]));
    if constexpr (HasMaxPool) {
      k_max_pool[pool_offset] = __float2half_rn(pool_max);
    }
  }
}

template <typename InputT>
__global__ void prepare_h3_sm89_v_single_load_kernel(
    const InputT* __restrict__ value,
    const int64_t* __restrict__ video_token_indices,
    const bool* __restrict__ video_slot_valid,
    half* __restrict__ packed_v,
    float* __restrict__ stage_amax,
    int64_t heads,
    int64_t prefix_tokens,
    int64_t prefix_blocks,
    int64_t key_physical_tokens,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_token) {
  constexpr int32_t HeadDim = 128;
  const int64_t physical_stage = blockIdx.x;
  const int64_t batch = blockIdx.z;
  const int64_t head = blockIdx.y;
  const int64_t head_batch = batch * heads + head;
  const int32_t channel = threadIdx.x;
  const bool is_prefix = physical_stage < prefix_blocks;
  const int64_t video_stage = physical_stage - prefix_blocks;
  __shared__ int64_t staged_token_indices[64];
  __shared__ bool staged_slot_valid[64];
  if (!is_prefix && channel < 64) {
    staged_token_indices[channel] =
        video_token_indices[video_stage * 64 + channel];
    staged_slot_valid[channel] =
        video_slot_valid[video_stage * 64 + channel];
  }
  __syncthreads();

  float amax = 0.0f;
#pragma unroll 1
  for (int32_t token = 0; token < 64; ++token) {
    const bool valid = is_prefix
        ? physical_stage * 64 + token < prefix_tokens
        : staged_slot_valid[token];
    const int64_t raw_token = is_prefix
        ? physical_stage * 64 + token
        : prefix_tokens + staged_token_indices[token];
    half narrowed = __float2half_rn(0.0f);
    if (valid) {
      narrowed = load_narrowed_scalar(
          value,
          batch * stride_batch + head * stride_head +
              raw_token * stride_token + channel);
    }
    const int64_t output_token = physical_stage * 64 + token;
    packed_v[(head_batch * key_physical_tokens + output_token) * HeadDim +
             channel] = narrowed;
    amax = fmaxf(amax, fabsf(__half2float(narrowed)));
  }
  stage_amax[
      (head_batch * (key_physical_tokens / 64) + physical_stage) * HeadDim +
      channel] = amax;
}

__global__ void prepare_h3_sm89_v_from_partials_kernel(
    const half* __restrict__ packed_v,
    const float* __restrict__ stage_amax,
    __nv_fp8_e4m3* __restrict__ v8,
    float* __restrict__ v_scale,
    int64_t heads,
    int64_t physical_stages,
    int64_t key_physical_tokens,
    int64_t padded_key_tokens) {
  constexpr int32_t HeadDim = 128;
  constexpr int32_t StageTokens = 64;
  constexpr int32_t Threads = 256;
  constexpr int32_t ChannelTile = 32;
  constexpr int32_t PackSize = 8;
  constexpr int32_t ChannelVectors = ChannelTile / PackSize;
  constexpr int32_t ReductionLanes = Threads / ChannelTile;
  constexpr int32_t StageStripes = 4;
  constexpr int32_t SharedPitch = 72;
  constexpr float ScaleMax = 2.25f;
  union __align__(16) SharedWorkspace {
    float reduction[Threads];
    half transposed[ChannelTile][SharedPitch];
  };
  __shared__ SharedWorkspace workspace;
  __shared__ float reciprocal[ChannelTile];

  const int32_t thread = threadIdx.x;
  const int32_t channel_tile = blockIdx.x / StageStripes;
  const int32_t stage_stripe = blockIdx.x % StageStripes;
  const int32_t head = blockIdx.y;
  const int32_t batch = blockIdx.z;
  const int32_t channel_base = channel_tile * ChannelTile;
  const int32_t local_channel = thread % ChannelTile;
  const int32_t reduction_lane = thread / ChannelTile;
  const int64_t head_batch = static_cast<int64_t>(batch) * heads + head;
  float thread_amax = 0.0f;
  const int64_t partial_base =
      head_batch * physical_stages * HeadDim + channel_base + local_channel;
  for (int64_t stage = reduction_lane; stage < physical_stages;
       stage += ReductionLanes) {
    thread_amax = fmaxf(
        thread_amax,
        stage_amax[partial_base + stage * HeadDim]);
  }
  workspace.reduction[thread] = thread_amax;
  __syncthreads();
  if (thread < ChannelTile) {
    float reduced_amax = 0.0f;
#pragma unroll
    for (int32_t lane_index = 0; lane_index < ReductionLanes; ++lane_index) {
      reduced_amax = fmaxf(
          reduced_amax,
          workspace.reduction[lane_index * ChannelTile + thread]);
    }
    reciprocal[thread] =
        reduced_amax == 0.0f ? 0.0f : ScaleMax / reduced_amax;
    if (stage_stripe == 0) {
      v_scale[head_batch * HeadDim + channel_base + thread] =
          reduced_amax / ScaleMax;
    }
  }
  __syncthreads();

  const int32_t natural_token = thread / ChannelVectors;
  const int32_t channel_vector = thread % ChannelVectors;
  const int32_t row_base = (natural_token / 16) * 16;
  const int32_t row_mod = natural_token % 16;
  const int32_t permuted_row =
      row_base + (row_mod / 8) * 2 + ((row_mod / 2) % 4) * 4 + row_mod % 2;
  const int32_t output_channel = thread / (StageTokens / PackSize);
  const int32_t output_token =
      (thread % (StageTokens / PackSize)) * PackSize;

#pragma unroll 1
  for (int64_t stage = stage_stripe;
       stage < padded_key_tokens / StageTokens;
       stage += StageStripes) {
    half values[PackSize];
    if (stage < physical_stages) {
      const int64_t input_offset =
          (head_batch * key_physical_tokens + stage * StageTokens +
           natural_token) * HeadDim +
          channel_base + channel_vector * PackSize;
      *reinterpret_cast<uint4*>(values) =
          *reinterpret_cast<const uint4*>(packed_v + input_offset);
    } else {
#pragma unroll
      for (int32_t element = 0; element < PackSize; ++element) {
        values[element] = __float2half_rn(0.0f);
      }
    }
#pragma unroll
    for (int32_t element = 0; element < PackSize; ++element) {
      workspace.transposed[channel_vector * PackSize + element][permuted_row] =
          values[element];
    }
    __syncthreads();
    *reinterpret_cast<uint4*>(values) = *reinterpret_cast<const uint4*>(
        workspace.transposed[output_channel] + output_token);
    const float multiplier = reciprocal[output_channel];
    float converted[PackSize];
#pragma unroll
    for (int32_t element = 0; element < PackSize; ++element) {
      converted[element] = multiplier == 0.0f
          ? 0.0f
          : __half2float(values[element]) * multiplier;
    }
    uint32_t fp8_words[2];
    floatx4_to_e4m3x4(fp8_words, converted, converted + 2);
    floatx4_to_e4m3x4(fp8_words + 1, converted + 4, converted + 6);
    const int64_t output_offset =
        (head_batch * HeadDim + channel_base + output_channel) *
            padded_key_tokens +
        stage * StageTokens + output_token;
    *reinterpret_cast<uint2*>(v8 + output_offset) =
        *reinterpret_cast<const uint2*>(fp8_words);
    __syncthreads();
  }
}

__global__ void reduce_h3_sm89_k_mean_kernel(
    const float* __restrict__ k_stage_sum,
    half* __restrict__ k_mean,
    int64_t heads,
    int64_t physical_stages,
    int64_t valid_tokens) {
  constexpr int32_t HeadDim = 128;
  const int32_t channel = threadIdx.x;
  const int64_t head_batch =
      static_cast<int64_t>(blockIdx.y) * heads + blockIdx.x;
  float sum = 0.0f;
  const int64_t base = head_batch * physical_stages * HeadDim + channel;
  for (int64_t stage = 0; stage < physical_stages; ++stage) {
    sum += k_stage_sum[base + stage * HeadDim];
  }
  k_mean[head_batch * HeadDim + channel] =
      __float2half_rn(sum / static_cast<float>(valid_tokens));
}

template <int32_t QueryBlock>
__global__ void quantize_h3_sm89_smoothed_k_kernel(
    const half* __restrict__ packed_k,
    const half* __restrict__ k_mean,
    const int32_t* __restrict__ video_valid_counts,
    int8_t* __restrict__ k8,
    float* __restrict__ k_scale,
    int64_t heads,
    int64_t prefix_tokens,
    int64_t prefix_blocks,
    int64_t physical_stages,
    int64_t key_physical_tokens) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  constexpr int32_t HeadDim = 128;
  constexpr int32_t Warps = HeadDim / 32;
  __shared__ half centered_tile[64 * HeadDim];
  __shared__ float warp_amax[Warps];
  __shared__ float block_scale;
  const int64_t physical_stage = blockIdx.x;
  const int64_t head_batch =
      static_cast<int64_t>(blockIdx.z) * heads + blockIdx.y;
  const int32_t channel = threadIdx.x;
  const int32_t lane = channel & 31;
  const int32_t warp = channel >> 5;
  int32_t valid_count;
  if (physical_stage < prefix_blocks) {
    valid_count = static_cast<int32_t>(min(
        static_cast<int64_t>(64),
        prefix_tokens - physical_stage * 64));
  } else {
    const int64_t video_stage = physical_stage - prefix_blocks;
    const int64_t logical_block =
        video_stage / static_cast<int64_t>(QueryBlock / 64);
    const int32_t half =
        static_cast<int32_t>(video_stage % (QueryBlock / 64));
    valid_count = max(
        0,
        min(64, video_valid_counts[logical_block] - half * 64));
  }
  const float mean = __half2float(k_mean[head_batch * HeadDim + channel]);
  float local_amax = 0.0f;
#pragma unroll
  for (int32_t token = 0; token < 64; ++token) {
    half centered = __float2half_rn(0.0f);
    if (token < valid_count) {
      const half raw = packed_k[
          (head_batch * key_physical_tokens + physical_stage * 64 + token) *
              HeadDim +
          channel];
      centered = __float2half_rn(__half2float(raw) - mean);
    }
    centered_tile[token * HeadDim + channel] = centered;
    local_amax = fmaxf(local_amax, fabsf(__half2float(centered)));
  }
  const float reduced = warp_allreduce_max(local_amax);
  if (lane == 0) warp_amax[warp] = reduced;
  __syncthreads();
  if (channel == 0) {
    float amax = warp_amax[0];
#pragma unroll
    for (int32_t warp_index = 1; warp_index < Warps; ++warp_index) {
      amax = fmaxf(amax, warp_amax[warp_index]);
    }
    block_scale = amax / 127.0f + 1.0e-7f;
    k_scale[head_batch * physical_stages + physical_stage] = block_scale;
  }
  __syncthreads();
#pragma unroll
  for (int32_t token = 0; token < 64; ++token) {
    float quantized =
        __half2float(centered_tile[token * HeadDim + channel]) / block_scale;
    quantized += quantized >= 0.0f ? 0.5f : -0.5f;
    k8[(head_batch * key_physical_tokens + physical_stage * 64 + token) *
           HeadDim +
       channel] = static_cast<int8_t>(__float2int_rz(quantized));
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

H3SM89Int8Prepared prepare_h3_sm89_int8_operands(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor video_token_indices,
    torch::Tensor video_slot_valid,
    torch::Tensor video_valid_counts,
    int64_t prefix_tokens,
    int64_t query_block_size,
    bool smooth_k,
    bool has_maxpool) {
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
      "SM89 H3 Q/K/V shapes must match");
  TORCH_CHECK(
      query.size(3) == 128,
      "SM89 fused INT8 preparation currently requires head_dim=128");
  TORCH_CHECK(
      prefix_tokens >= 0 && prefix_tokens < query.size(2),
      "prefix_tokens must be nonnegative and smaller than the sequence");
  TORCH_CHECK(
      query_block_size == 64 || query_block_size == 128,
      "query_block_size must be 64 or 128");
  TORCH_CHECK(
      video_token_indices.is_cuda() &&
          video_token_indices.device() == query.device() &&
          video_token_indices.scalar_type() == at::ScalarType::Long &&
          video_token_indices.dim() == 1 &&
          video_token_indices.is_contiguous(),
      "video_token_indices must be contiguous CUDA int64");
  TORCH_CHECK(
      video_slot_valid.is_cuda() &&
          video_slot_valid.device() == query.device() &&
          video_slot_valid.scalar_type() == at::ScalarType::Bool &&
          video_slot_valid.dim() == 1 && video_slot_valid.is_contiguous() &&
          video_slot_valid.numel() == video_token_indices.numel(),
      "video_slot_valid must be matching contiguous CUDA bool");
  TORCH_CHECK(
      video_valid_counts.is_cuda() &&
          video_valid_counts.device() == query.device() &&
          video_valid_counts.scalar_type() == at::ScalarType::Int &&
          video_valid_counts.dim() == 1 &&
          video_valid_counts.is_contiguous(),
      "video_valid_counts must be contiguous CUDA int32");
  const int64_t video_physical_tokens = video_token_indices.numel();
  TORCH_CHECK(
      video_physical_tokens > 0 &&
          video_physical_tokens % query_block_size == 0,
      "video physical capacity must be a positive multiple of query_block_size");
  const int64_t video_blocks = video_physical_tokens / query_block_size;
  TORCH_CHECK(
      video_valid_counts.numel() == video_blocks,
      "video_valid_counts must have one item per query block");
  const int64_t prefix_blocks = (prefix_tokens + 63) / 64;
  const int64_t key_physical_tokens =
      prefix_blocks * 64 + video_physical_tokens;
  const int64_t physical_stages = key_physical_tokens / 64;
  const int64_t padded_key_tokens = ((key_physical_tokens + 127) / 128) * 128;
  check_raster_and_grid(
      query, 2 * video_blocks + prefix_blocks, query.size(1));

  c10::cuda::CUDAGuard device_guard(query.device());
  const auto fp16_options = query.options().dtype(at::ScalarType::Half);
  const auto int8_options = query.options().dtype(at::ScalarType::Char);
  const auto fp32_options = query.options().dtype(at::ScalarType::Float);
  auto q_pool = torch::empty(
      {query.size(0), query.size(1), video_blocks, 128}, fp16_options);
  auto k_pool = torch::empty_like(q_pool);
  auto q_max_pool = has_maxpool ? torch::empty_like(q_pool)
                                : torch::empty({0}, fp16_options);
  auto k_max_pool = has_maxpool ? torch::empty_like(q_pool)
                                : torch::empty({0}, fp16_options);
  auto packed_q = torch::empty(
      {query.size(0), query.size(1), video_physical_tokens, 128},
      fp16_options);
  auto packed_k = torch::empty(
      {query.size(0), query.size(1), key_physical_tokens, 128},
      fp16_options);
  auto packed_v = torch::empty_like(packed_k);
  auto q8 = torch::empty(packed_q.sizes(), int8_options);
  auto k8 = torch::empty(packed_k.sizes(), int8_options);
  auto v8 = torch::empty(
      {query.size(0), query.size(1), 128, padded_key_tokens},
      query.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto q_scale = torch::empty(
      {query.size(0), query.size(1), video_blocks}, fp32_options);
  auto k_scale = torch::empty(
      {query.size(0), query.size(1), physical_stages}, fp32_options);
  auto v_scale = torch::empty(
      {query.size(0), query.size(1), 128}, fp32_options);
  auto k_mean = smooth_k
      ? torch::empty({query.size(0), query.size(1), 128}, fp16_options)
      : torch::empty({0}, fp16_options);
  auto k_stage_sum = smooth_k
      ? torch::empty(
            {query.size(0), query.size(1), physical_stages, 128},
            fp32_options)
      : torch::empty({0}, fp32_options);
  auto v_stage_amax = torch::empty(
      {query.size(0), query.size(1), physical_stages, 128},
      fp32_options);

  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(query.get_device());
  const dim3 v_grid(
      static_cast<uint32_t>(physical_stages),
      static_cast<uint32_t>(query.size(1)),
      static_cast<uint32_t>(query.size(0)));
  const auto launch = [&](auto query_block_tag, auto* input_tag) {
    constexpr int32_t QueryBlock = decltype(query_block_tag)::value;
    using InputT = std::remove_pointer_t<decltype(input_tag)>;
    const auto launch_qk = [&](auto smooth_tag, auto maxpool_tag) {
      constexpr bool SmoothK = decltype(smooth_tag)::value;
      constexpr bool HasMaxPool = decltype(maxpool_tag)::value;
      const dim3 fused_qk_grid(
          static_cast<uint32_t>(QueryBlock == 128
              ? video_blocks
              : 2 * video_blocks + prefix_blocks),
          static_cast<uint32_t>(query.size(1)),
          static_cast<uint32_t>(query.size(0)));
      prepare_h3_sm89_qk_single_load_kernel<
          InputT, QueryBlock, SmoothK, HasMaxPool><<<
          fused_qk_grid, 128, 0, stream>>>(
          reinterpret_cast<const InputT*>(query.data_ptr()),
          reinterpret_cast<const InputT*>(key.data_ptr()),
          video_token_indices.data_ptr<int64_t>(),
          video_slot_valid.data_ptr<bool>(),
          video_valid_counts.data_ptr<int32_t>(),
          reinterpret_cast<half*>(q_pool.data_ptr<at::Half>()),
          reinterpret_cast<half*>(k_pool.data_ptr<at::Half>()),
          HasMaxPool
              ? reinterpret_cast<half*>(q_max_pool.data_ptr<at::Half>())
              : nullptr,
          HasMaxPool
              ? reinterpret_cast<half*>(k_max_pool.data_ptr<at::Half>())
              : nullptr,
          reinterpret_cast<half*>(packed_q.data_ptr<at::Half>()),
          reinterpret_cast<half*>(packed_k.data_ptr<at::Half>()),
          q8.data_ptr<int8_t>(),
          k8.data_ptr<int8_t>(),
          q_scale.data_ptr<float>(),
          k_scale.data_ptr<float>(),
          SmoothK ? k_stage_sum.data_ptr<float>() : nullptr,
          query.size(1), video_blocks, prefix_tokens, prefix_blocks,
          video_physical_tokens, key_physical_tokens,
          query.stride(0), query.stride(1), query.stride(2),
          key.stride(0), key.stride(1), key.stride(2));
      if constexpr (QueryBlock == 128) {
        const dim3 key_grid(
            static_cast<uint32_t>(video_blocks + prefix_blocks),
            static_cast<uint32_t>(query.size(1)),
            static_cast<uint32_t>(query.size(0)));
        prepare_h3_sm89_q128_key_single_load_kernel<
            InputT, SmoothK, HasMaxPool><<<key_grid, 128, 0, stream>>>(
            reinterpret_cast<const InputT*>(key.data_ptr()),
            video_token_indices.data_ptr<int64_t>(),
            video_slot_valid.data_ptr<bool>(),
            video_valid_counts.data_ptr<int32_t>(),
            reinterpret_cast<half*>(k_pool.data_ptr<at::Half>()),
            HasMaxPool
                ? reinterpret_cast<half*>(k_max_pool.data_ptr<at::Half>())
                : nullptr,
            reinterpret_cast<half*>(packed_k.data_ptr<at::Half>()),
            k8.data_ptr<int8_t>(), k_scale.data_ptr<float>(),
            SmoothK ? k_stage_sum.data_ptr<float>() : nullptr,
            query.size(1), video_blocks, prefix_tokens, prefix_blocks,
            key_physical_tokens,
            key.stride(0), key.stride(1), key.stride(2));
      }
    };
    if (smooth_k) {
      if (has_maxpool) {
        launch_qk(std::true_type{}, std::true_type{});
      } else {
        launch_qk(std::true_type{}, std::false_type{});
      }
    } else if (has_maxpool) {
      launch_qk(std::false_type{}, std::true_type{});
    } else {
      launch_qk(std::false_type{}, std::false_type{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    prepare_h3_sm89_v_single_load_kernel<InputT><<<v_grid, 128, 0, stream>>>(
        reinterpret_cast<const InputT*>(value.data_ptr()),
        video_token_indices.data_ptr<int64_t>(),
        video_slot_valid.data_ptr<bool>(),
        reinterpret_cast<half*>(packed_v.data_ptr<at::Half>()),
        v_stage_amax.data_ptr<float>(),
        value.size(1), prefix_tokens, prefix_blocks, key_physical_tokens,
        value.stride(0), value.stride(1), value.stride(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (smooth_k) {
      const dim3 mean_grid(
          static_cast<uint32_t>(key.size(1)),
          static_cast<uint32_t>(key.size(0)));
      reduce_h3_sm89_k_mean_kernel<<<mean_grid, 128, 0, stream>>>(
          k_stage_sum.data_ptr<float>(),
          reinterpret_cast<half*>(k_mean.data_ptr<at::Half>()),
          key.size(1), physical_stages, key.size(2));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      quantize_h3_sm89_smoothed_k_kernel<QueryBlock><<<
          v_grid, 128, 0, stream>>>(
          reinterpret_cast<const half*>(packed_k.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(k_mean.data_ptr<at::Half>()),
          video_valid_counts.data_ptr<int32_t>(),
          k8.data_ptr<int8_t>(), k_scale.data_ptr<float>(),
          key.size(1), prefix_tokens, prefix_blocks, physical_stages,
          key_physical_tokens);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  const auto dispatch_input = [&](auto* input_tag) {
    if (query_block_size == 64) {
      launch(std::integral_constant<int32_t, 64>{}, input_tag);
    } else {
      launch(std::integral_constant<int32_t, 128>{}, input_tag);
    }
  };
  if (query.scalar_type() == at::ScalarType::Half) {
    dispatch_input(static_cast<half*>(nullptr));
  } else {
    dispatch_input(static_cast<nv_bfloat16*>(nullptr));
  }

  constexpr int32_t stage_stripes = 4;
  constexpr int32_t channel_tiles = 128 / 32;
  const dim3 v_quant_grid(
      channel_tiles * stage_stripes,
      static_cast<uint32_t>(value.size(1)),
      static_cast<uint32_t>(value.size(0)));
  prepare_h3_sm89_v_from_partials_kernel<<<
      v_quant_grid, 256, 0, stream>>>(
      reinterpret_cast<const half*>(packed_v.data_ptr<at::Half>()),
      v_stage_amax.data_ptr<float>(),
      reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
      v_scale.data_ptr<float>(),
      value.size(1), physical_stages, key_physical_tokens,
      padded_key_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      q_pool, k_pool, packed_q, packed_k, packed_v, q8, k8, v8,
      q_scale, k_scale, v_scale, k_mean, q_max_pool, k_max_pool};
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
