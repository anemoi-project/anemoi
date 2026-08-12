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
 * Project-owned Draft-Q/K probability implementation. cuBLAS materializes
 * scaled FP16 logits from FP16 operands with FP32 Tensor Core accumulation;
 * the row softmax below keeps max, sum, and exp arithmetic in FP32 and rounds
 * only the final probability to FP16.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "../api.h"

namespace {

constexpr int kSoftmaxThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kSoftmaxWarps = kSoftmaxThreads / kWarpSize;
constexpr int64_t kMaxGridX = 2147483647LL;
constexpr cublasComputeType_t kDraftComputeType = CUBLAS_COMPUTE_32F;
constexpr cublasGemmAlgo_t kDraftAlgorithm = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
// Routing predicts the unshrunk Gaussian/Jensen pooling error.  The attention
// epilogue intentionally keeps its separately calibrated lambda=0.5
// compensation (0.25*sigma^2); selection uses the full 0.5*sigma^2 moment so
// blocks whose surrogate is intrinsically difficult receive exact compute.
constexpr float kJensenRiskCoefficient = 0.5f;
constexpr float kMaximumJensenBias = 10.0f;
constexpr float kMaximumHalf = 65504.0f;
constexpr int kRasterPatchRows = 8;
constexpr int kRasterPatchColumns = 16;
constexpr int kRasterPatchTokens =
    kRasterPatchRows * kRasterPatchColumns;
constexpr int kSpatialQueryGroups = 4;
constexpr int kSpatialKeyGroups = 8;
constexpr int kSpatialValueSketch = 8;
constexpr int kSpatialHeadDim = 128;

inline int64_t checked_positive_product(
    int64_t lhs, int64_t rhs, const char* description) {
  TORCH_CHECK(lhs > 0 && rhs > 0, description, " factors must be positive");
  TORCH_CHECK(
      lhs <= std::numeric_limits<int64_t>::max() / rhs,
      description, " exceeds int64 range");
  return lhs * rhs;
}

inline void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      operation, " failed with cuBLAS status ", static_cast<int>(status));
}

inline void check_pool_tensor(
    const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be a defined tensor");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous BHSD");
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Half,
      name, " must have dtype torch.float16");
  TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B,H,R,D]");
  TORCH_CHECK(
      tensor.size(0) > 0 && tensor.size(1) > 0 && tensor.size(2) > 0,
      name, " batch, head, and row dimensions must be positive");
  TORCH_CHECK(
      tensor.size(3) == 64 || tensor.size(3) == 128,
      name, " head dimension must be 64 or 128");
}

inline void check_skip_risk_tensors(
    const torch::Tensor& q_pool,
    const torch::Tensor& k_pool,
    const torch::Tensor& qk_patch_statistics,
    const torch::Tensor& v_sum) {
  const int64_t batch = q_pool.size(0);
  const int64_t q_heads = q_pool.size(1);
  const int64_t kv_heads = k_pool.size(1);
  const int64_t rows = q_pool.size(2);
  const int64_t head_dim = q_pool.size(3);
  TORCH_CHECK(
      qk_patch_statistics.defined() && qk_patch_statistics.is_cuda() &&
          qk_patch_statistics.is_contiguous(),
      "qk_patch_statistics must be a contiguous CUDA tensor");
  TORCH_CHECK(
      qk_patch_statistics.scalar_type() == at::ScalarType::Half,
      "qk_patch_statistics must have dtype torch.float16");
  TORCH_CHECK(
      qk_patch_statistics.sizes() ==
          torch::IntArrayRef({batch, q_heads + kv_heads, rows}),
      "qk_patch_statistics must have shape [B,Hq+Hkv,R]");
  TORCH_CHECK(
      qk_patch_statistics.device() == q_pool.device(),
      "qk_patch_statistics must share the pooled-Q/K CUDA device");
  TORCH_CHECK(
      v_sum.defined() && v_sum.is_cuda() && v_sum.is_contiguous(),
      "v_sum must be a contiguous CUDA tensor");
  TORCH_CHECK(
      v_sum.scalar_type() == at::ScalarType::Float,
      "v_sum must have dtype torch.float32");
  TORCH_CHECK(
      v_sum.sizes() ==
          torch::IntArrayRef({batch, kv_heads, rows, head_dim}),
      "v_sum must have shape [B,Hkv,R,D]");
  TORCH_CHECK(
      v_sum.device() == q_pool.device(),
      "v_sum must share the pooled-Q/K CUDA device");
}

inline int64_t checked_patch_count(
    int64_t frames, int64_t height, int64_t width) {
  TORCH_CHECK(frames > 0 && height > 0 && width > 0,
              "frames, height, and width must be positive");
  const int64_t patch_rows = (height + 7) / 8;
  const int64_t patch_columns = (width + 15) / 16;
  return checked_positive_product(
      frames,
      checked_positive_product(
          patch_rows, patch_columns, "raster patches per frame"),
      "raster patch count");
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <bool IsMax>
__device__ __forceinline__ float block_reduce(
    float value, float* __restrict__ warp_scratch) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  value = IsMax ? warp_max(value) : warp_sum(value);
  if (lane == 0) {
    warp_scratch[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = lane < kSoftmaxWarps
        ? warp_scratch[lane]
        : (IsMax ? -CUDART_INF_F : 0.0f);
    value = IsMax ? warp_max(value) : warp_sum(value);
    if (lane == 0) {
      warp_scratch[0] = value;
    }
  }
  __syncthreads();
  return warp_scratch[0];
}

__global__ __launch_bounds__(kSoftmaxThreads) void row_softmax_fp16_kernel(
    half* logits_probability,
    float* row_lse,
    int64_t row_count,
    int64_t columns) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int64_t row_offset = row * columns;
  __shared__ float warp_scratch[kSoftmaxWarps];

  float local_max = -CUDART_INF_F;
  for (int64_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    local_max = fmaxf(
        local_max, __half2float(logits_probability[row_offset + column]));
  }
  const float row_max = block_reduce<true>(local_max, warp_scratch);

  float local_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    local_sum += expf(
        __half2float(logits_probability[row_offset + column]) - row_max);
  }
  const float row_sum = block_reduce<false>(local_sum, warp_scratch);
  const float inverse_sum = 1.0f / row_sum;
  if (row_lse != nullptr && threadIdx.x == 0) {
    row_lse[row] = row_max + logf(row_sum);
  }

  for (int64_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const float normalized = expf(
        __half2float(logits_probability[row_offset + column]) - row_max) *
        inverse_sum;
    logits_probability[row_offset + column] = __float2half_rn(normalized);
  }
}

__device__ __forceinline__ int valid_patch_tokens(
    int patch,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int spatial = patch % (patch_rows * patch_columns);
  const int patch_row = spatial / patch_columns;
  const int patch_column = spatial - patch_row * patch_columns;
  const int valid_rows = min(8, height - patch_row * 8);
  const int valid_columns = min(16, width - patch_column * 16);
  return valid_rows * valid_columns;
}

__device__ __forceinline__ int valid_k64_tokens(
    int key_stage,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int patch = key_stage >> 1;
  const int half = key_stage & 1;
  const int spatial = patch % (patch_rows * patch_columns);
  const int patch_row = spatial / patch_columns;
  const int patch_column = spatial - patch_row * patch_columns;
  const int valid_rows = min(8, height - patch_row * 8);
  const int valid_columns = min(16, width - patch_column * 16);
  return max(0, min(4, valid_rows - half * 4)) * valid_columns;
}

__device__ __forceinline__ int valid_spatial_group_tokens(
    int patch,
    int group,
    int row_groups,
    int column_groups,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int spatial = patch % (patch_rows * patch_columns);
  const int patch_row = spatial / patch_columns;
  const int patch_column = spatial - patch_row * patch_columns;
  const int valid_rows = min(
      kRasterPatchRows, height - patch_row * kRasterPatchRows);
  const int valid_columns = min(
      kRasterPatchColumns, width - patch_column * kRasterPatchColumns);
  const int group_row = group / column_groups;
  const int group_column = group - group_row * column_groups;
  const int rows_per_group = kRasterPatchRows / row_groups;
  const int columns_per_group = kRasterPatchColumns / column_groups;
  const int group_rows = max(
      0, min(rows_per_group, valid_rows - group_row * rows_per_group));
  const int group_columns = max(
      0,
      min(
          columns_per_group,
          valid_columns - group_column * columns_per_group));
  return group_rows * group_columns;
}

__device__ __forceinline__ int hadamard_sketch_column(int sketch) {
  switch (sketch) {
    case 0: return 3;
    case 1: return 17;
    case 2: return 29;
    case 3: return 43;
    case 4: return 67;
    case 5: return 83;
    case 6: return 101;
    default: return 127;
  }
}

// Logical input is contiguous [B,H,R*128,D128], where the 128-token extent
// preserves physical 8x16 raster slots and padded edge slots are ignored.
// Logical output is contiguous [B,H,R*4,D128]: 2x2 spatial groups, each a
// 4x8 rectangle before edge clipping.  One CTA owns one (B,H,patch).
__global__ __launch_bounds__(kSpatialHeadDim)
void q4_spatial_centroid_kernel(
    const half* __restrict__ packed_q,
    half* __restrict__ q_centroid,
    int patches,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int64_t owner_patch = static_cast<int64_t>(blockIdx.x);
  const int patch = static_cast<int>(owner_patch % patches);
  const int dimension = static_cast<int>(threadIdx.x);
  const int spatial = patch % (patch_rows * patch_columns);
  const int patch_row = spatial / patch_columns;
  const int patch_column = spatial - patch_row * patch_columns;
  const int valid_rows = min(
      kRasterPatchRows, height - patch_row * kRasterPatchRows);
  const int valid_columns = min(
      kRasterPatchColumns, width - patch_column * kRasterPatchColumns);
  const int64_t input_base =
      owner_patch * kRasterPatchTokens * kSpatialHeadDim;
  const int64_t output_base =
      owner_patch * kSpatialQueryGroups * kSpatialHeadDim;
  float sums[kSpatialQueryGroups] = {};
  for (int row = 0; row < valid_rows; ++row) {
    for (int column = 0; column < valid_columns; ++column) {
      const int group = (row >> 2) * 2 + (column >> 3);
      const int slot = row * kRasterPatchColumns + column;
      sums[group] += __half2float(
          packed_q[input_base +
                   static_cast<int64_t>(slot) * kSpatialHeadDim +
                   dimension]);
    }
  }
#pragma unroll
  for (int group = 0; group < kSpatialQueryGroups; ++group) {
    const int count = valid_spatial_group_tokens(
        patch, group, 2, 2, patch_rows, patch_columns, height, width);
    q_centroid[
        output_base + static_cast<int64_t>(group) * kSpatialHeadDim +
        dimension] = __float2half_rn(
            count > 0 ? sums[group] / static_cast<float>(count) : 0.0f);
  }
}

// K uses 4x2 groups (nominally 2x8 tokens).  V is reduced over the identical
// groups and projected onto eight fixed, orthogonal Hadamard columns.  The
// stored [B,Hkv,R*8,8] sketch is centered by the count-weighted patch mean,
// so the scoring kernel needs neither a second coarse probability map nor a
// full-D approximate output vector.
__global__ __launch_bounds__(kSpatialHeadDim)
void k8_centroid_centered_vsketch_kernel(
    const half* __restrict__ packed_k,
    const half* __restrict__ packed_v,
    half* __restrict__ k_centroid,
    half* __restrict__ centered_v_sketch,
    int patches,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int64_t owner_patch = static_cast<int64_t>(blockIdx.x);
  const int patch = static_cast<int>(owner_patch % patches);
  const int dimension = static_cast<int>(threadIdx.x);
  const int spatial = patch % (patch_rows * patch_columns);
  const int patch_row = spatial / patch_columns;
  const int patch_column = spatial - patch_row * patch_columns;
  const int valid_rows = min(
      kRasterPatchRows, height - patch_row * kRasterPatchRows);
  const int valid_columns = min(
      kRasterPatchColumns, width - patch_column * kRasterPatchColumns);
  const int64_t input_base =
      owner_patch * kRasterPatchTokens * kSpatialHeadDim;
  const int64_t k_output_base =
      owner_patch * kSpatialKeyGroups * kSpatialHeadDim;
  float k_sums[kSpatialKeyGroups] = {};
  float v_sums[kSpatialKeyGroups] = {};
  for (int row = 0; row < valid_rows; ++row) {
    for (int column = 0; column < valid_columns; ++column) {
      const int group = (row >> 1) * 2 + (column >> 3);
      const int slot = row * kRasterPatchColumns + column;
      const int64_t offset =
          input_base + static_cast<int64_t>(slot) * kSpatialHeadDim +
          dimension;
      k_sums[group] += __half2float(packed_k[offset]);
      v_sums[group] += __half2float(packed_v[offset]);
    }
  }

  __shared__ float v_group_mean[kSpatialKeyGroups][kSpatialHeadDim];
  __shared__ float v_group_projection[kSpatialKeyGroups][kSpatialValueSketch];
#pragma unroll
  for (int group = 0; group < kSpatialKeyGroups; ++group) {
    const int count = valid_spatial_group_tokens(
        patch, group, 4, 2, patch_rows, patch_columns, height, width);
    const float inverse_count = count > 0 ? 1.0f / count : 0.0f;
    k_centroid[
        k_output_base + static_cast<int64_t>(group) * kSpatialHeadDim +
        dimension] = __float2half_rn(k_sums[group] * inverse_count);
    v_group_mean[group][dimension] = v_sums[group] * inverse_count;
  }
  __syncthreads();

  if (dimension < kSpatialKeyGroups * kSpatialValueSketch) {
    const int group = dimension / kSpatialValueSketch;
    const int sketch = dimension - group * kSpatialValueSketch;
    const int column = hadamard_sketch_column(sketch);
    float projected = 0.0f;
#pragma unroll
    for (int channel = 0; channel < kSpatialHeadDim; ++channel) {
      const float sign = (__popc(channel & column) & 1) ? -1.0f : 1.0f;
      projected += v_group_mean[group][channel] * sign;
    }
    v_group_projection[group][sketch] =
        projected * 0.3535533905932738f;
  }
  __syncthreads();

  if (dimension < kSpatialKeyGroups * kSpatialValueSketch) {
    const int group = dimension / kSpatialValueSketch;
    const int sketch = dimension - group * kSpatialValueSketch;
    float patch_projection = 0.0f;
    int patch_count = 0;
#pragma unroll
    for (int source_group = 0; source_group < kSpatialKeyGroups;
         ++source_group) {
      const int count = valid_spatial_group_tokens(
          patch, source_group, 4, 2,
          patch_rows, patch_columns, height, width);
      patch_projection +=
          v_group_projection[source_group][sketch] * count;
      patch_count += count;
    }
    patch_projection /= static_cast<float>(patch_count);
    const int64_t output_base =
        owner_patch * kSpatialKeyGroups * kSpatialValueSketch;
    centered_v_sketch[
        output_base + static_cast<int64_t>(group) * kSpatialValueSketch +
        sketch] = __float2half_rn(
            v_group_projection[group][sketch] - patch_projection);
  }
}

__global__ __launch_bounds__(kSoftmaxThreads)
void row_spatial_probability_fp16_kernel(
    half* __restrict__ logits_probability,
    int64_t row_count,
    int columns,
    int patches,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int64_t row_offset = row * columns;
  __shared__ float warp_scratch[kSoftmaxWarps];
  float local_max = -CUDART_INF_F;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const int key_patch = column / kSpatialKeyGroups;
    const int key_group = column - key_patch * kSpatialKeyGroups;
    const int count = valid_spatial_group_tokens(
        key_patch, key_group, 4, 2,
        patch_rows, patch_columns, height, width);
    const float adjusted = count > 0
        ? __half2float(logits_probability[row_offset + column]) +
              logf(static_cast<float>(count))
        : -CUDART_INF_F;
    local_max = fmaxf(local_max, adjusted);
  }
  const float row_max = block_reduce<true>(local_max, warp_scratch);
  float local_sum = 0.0f;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const int key_patch = column / kSpatialKeyGroups;
    const int key_group = column - key_patch * kSpatialKeyGroups;
    const int count = valid_spatial_group_tokens(
        key_patch, key_group, 4, 2,
        patch_rows, patch_columns, height, width);
    if (count > 0) {
      const float adjusted =
          __half2float(logits_probability[row_offset + column]) +
          logf(static_cast<float>(count));
      local_sum += expf(adjusted - row_max);
    }
  }
  const float inverse_sum =
      1.0f / block_reduce<false>(local_sum, warp_scratch);
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const int key_patch = column / kSpatialKeyGroups;
    const int key_group = column - key_patch * kSpatialKeyGroups;
    const int count = valid_spatial_group_tokens(
        key_patch, key_group, 4, 2,
        patch_rows, patch_columns, height, width);
    float probability = 0.0f;
    if (count > 0) {
      const float adjusted =
          __half2float(logits_probability[row_offset + column]) +
          logf(static_cast<float>(count));
      probability = expf(adjusted - row_max) * inverse_sum;
    }
    logits_probability[row_offset + column] =
        __float2half_rn(probability);
  }
}

// One CTA owns one (B,Hq,query-patch); each warp streams a disjoint subset of
// key patches.  Its 32 lanes are four Q groups x eight V-sketch coordinates.
// Every fine probability is read once per sketch coordinate from L1, while
// the compact centered V sketch remains resident-cache friendly.
__global__ __launch_bounds__(kSoftmaxThreads)
void spatial_residual_score_kernel(
    const half* __restrict__ probability,
    const half* __restrict__ centered_v_sketch,
    const half* __restrict__ base_risk,
    half* __restrict__ output_risk,
    int patches,
    int q_heads,
    int kv_heads,
    int patch_rows,
    int patch_columns,
    int height,
    int width,
    float residual_coefficient) {
  const int64_t query_owner = static_cast<int64_t>(blockIdx.x);
  const int query_patch = static_cast<int>(query_owner % patches);
  const int q_head = static_cast<int>((query_owner / patches) % q_heads);
  const int batch = static_cast<int>(
      query_owner / (static_cast<int64_t>(q_heads) * patches));
  const int kv_head = q_head / (q_heads / kv_heads);
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int query_group = lane / kSpatialValueSketch;
  const int sketch = lane - query_group * kSpatialValueSketch;
  const int64_t probability_head_base =
      (static_cast<int64_t>(batch) * q_heads + q_head) *
      (static_cast<int64_t>(patches) * kSpatialQueryGroups) *
      (static_cast<int64_t>(patches) * kSpatialKeyGroups);
  const int64_t probability_row =
      probability_head_base +
      (static_cast<int64_t>(query_patch) * kSpatialQueryGroups +
       query_group) *
          (static_cast<int64_t>(patches) * kSpatialKeyGroups);
  const int64_t sketch_head_base =
      (static_cast<int64_t>(batch) * kv_heads + kv_head) * patches *
      kSpatialKeyGroups * kSpatialValueSketch;
  const int64_t risk_head_base =
      (static_cast<int64_t>(batch) * q_heads + q_head) * patches * patches;
  const int query_count = valid_patch_tokens(
      query_patch, patch_rows, patch_columns, height, width);

  for (int key_patch = warp; key_patch < patches;
       key_patch += kSoftmaxWarps) {
    float residual = 0.0f;
#pragma unroll
    for (int key_group = 0; key_group < kSpatialKeyGroups; ++key_group) {
      const float mass = __half2float(
          probability[
              probability_row +
              static_cast<int64_t>(key_patch) * kSpatialKeyGroups +
              key_group]);
      const float value = __half2float(
          centered_v_sketch[
              sketch_head_base +
              (static_cast<int64_t>(key_patch) * kSpatialKeyGroups +
               key_group) *
                  kSpatialValueSketch +
              sketch]);
      residual = fmaf(mass, value, residual);
    }
    float squared = residual * residual;
#pragma unroll
    for (int offset = kSpatialValueSketch / 2; offset > 0; offset /= 2) {
      squared += __shfl_down_sync(
          0xffffffffu, squared, offset, kSpatialValueSketch);
    }
    float weighted_squared = 0.0f;
    if (sketch == 0) {
      const int group_count = valid_spatial_group_tokens(
          query_patch, query_group, 2, 2,
          patch_rows, patch_columns, height, width);
      weighted_squared = squared * group_count;
    }
    weighted_squared = warp_sum(weighted_squared);
    if (lane == 0) {
      const float residual_norm = sqrtf(
          fmaxf(weighted_squared / static_cast<float>(query_count), 0.0f));
      const int64_t risk_offset =
          risk_head_base + static_cast<int64_t>(query_patch) * patches +
          key_patch;
      const float base = fmaxf(__half2float(base_risk[risk_offset]), 0.0f);
      const float scaled = residual_coefficient * residual_norm;
      const float combined = sqrtf(fmaf(base, base, scaled * scaled));
      output_risk[risk_offset] =
          __float2half_rn(fminf(combined, kMaximumHalf));
    }
  }
}

__global__ void global_v_mean_kernel(
    const float* __restrict__ v_sum,
    float* __restrict__ global_v_mean,
    int rows,
    int head_dim,
    int video_tokens) {
  const int owner = static_cast<int>(blockIdx.x);
  const int64_t owner_offset = static_cast<int64_t>(owner) * rows * head_dim;
  const int64_t output_offset = static_cast<int64_t>(owner) * head_dim;
  for (int dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    float sum = 0.0f;
    for (int patch = 0; patch < rows; ++patch) {
      sum += v_sum[owner_offset + static_cast<int64_t>(patch) * head_dim +
                   dimension];
    }
    global_v_mean[output_offset + dimension] =
        sum / static_cast<float>(video_tokens);
  }
}

__global__ __launch_bounds__(kSoftmaxThreads) void v_sensitivity_kernel(
    const float* __restrict__ v_sum,
    const float* __restrict__ global_v_mean,
    float* __restrict__ v_sensitivity,
    int rows,
    int head_dim,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int64_t linear_patch = static_cast<int64_t>(blockIdx.x);
  const int patch = static_cast<int>(linear_patch % rows);
  const int owner = static_cast<int>(linear_patch / rows);
  const int count = valid_patch_tokens(
      patch, patch_rows, patch_columns, height, width);
  const int64_t patch_offset = linear_patch * head_dim;
  const int64_t mean_offset = static_cast<int64_t>(owner) * head_dim;
  __shared__ float warp_scratch[kSoftmaxWarps];
  float sum_squares = 0.0f;
  for (int dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    const float patch_mean =
        v_sum[patch_offset + dimension] / static_cast<float>(count);
    const float difference = patch_mean - global_v_mean[mean_offset + dimension];
    sum_squares += difference * difference;
  }
  const float total = block_reduce<false>(sum_squares, warp_scratch);
  if (threadIdx.x == 0) {
    v_sensitivity[linear_patch] = sqrtf(fmaxf(total, 0.0f));
  }
}

__global__ __launch_bounds__(kSoftmaxThreads) void v_sensitivity_k64_kernel(
    const float* __restrict__ v_sum,
    const float* __restrict__ global_v_mean,
    float* __restrict__ v_sensitivity,
    int key_groups,
    int head_dim,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int64_t linear_group = static_cast<int64_t>(blockIdx.x);
  const int key_stage = static_cast<int>(linear_group % key_groups);
  const int owner = static_cast<int>(linear_group / key_groups);
  const int count = valid_k64_tokens(
      key_stage, patch_rows, patch_columns, height, width);
  const int64_t group_offset = linear_group * head_dim;
  const int64_t mean_offset = static_cast<int64_t>(owner) * head_dim;
  __shared__ float warp_scratch[kSoftmaxWarps];
  float sum_squares = 0.0f;
  for (int dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    const float group_mean =
        v_sum[group_offset + dimension] / static_cast<float>(count);
    const float difference = group_mean - global_v_mean[mean_offset + dimension];
    sum_squares += difference * difference;
  }
  const float total = block_reduce<false>(sum_squares, warp_scratch);
  if (threadIdx.x == 0) {
    v_sensitivity[linear_group] = sqrtf(fmaxf(total, 0.0f));
  }
}

__global__ __launch_bounds__(kSoftmaxThreads) void row_skip_error_risk_kernel(
    half* __restrict__ logits_risk,
    const half* __restrict__ qk_patch_statistics,
    const float* __restrict__ v_sensitivity,
    int64_t row_count,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int query_patch = static_cast<int>(row % rows);
  const int q_head = static_cast<int>((row / rows) % q_heads);
  const int batch = static_cast<int>(row / (static_cast<int64_t>(q_heads) * rows));
  const int queries_per_kv = q_heads / kv_heads;
  const int kv_head = q_head / queries_per_kv;
  const int statistics_heads = q_heads + kv_heads;
  const int64_t q_statistics_offset =
      (static_cast<int64_t>(batch) * statistics_heads + q_head) * rows +
      query_patch;
  const int64_t k_statistics_offset =
      (static_cast<int64_t>(batch) * statistics_heads + q_heads + kv_head) *
      rows;
  const int64_t sensitivity_offset =
      (static_cast<int64_t>(batch) * kv_heads + kv_head) * rows;
  const int64_t row_offset = row * rows;
  const float q_energy =
      fmaxf(__half2float(qk_patch_statistics[q_statistics_offset]), 0.0f);
  __shared__ float warp_scratch[kSoftmaxWarps];

  float local_max = -CUDART_INF_F;
  for (int key_patch = threadIdx.x; key_patch < rows;
       key_patch += blockDim.x) {
    const float k_variance = fmaxf(
        __half2float(qk_patch_statistics[k_statistics_offset + key_patch]),
        0.0f);
    const float bias = fminf(
        kJensenRiskCoefficient * q_energy * k_variance /
            static_cast<float>(head_dim),
        kMaximumJensenBias);
    const int count = valid_patch_tokens(
        key_patch, patch_rows, patch_columns, height, width);
    const float adjusted =
        __half2float(logits_risk[row_offset + key_patch]) +
        logf(static_cast<float>(count)) + bias;
    local_max = fmaxf(local_max, adjusted);
  }
  const float row_max = block_reduce<true>(local_max, warp_scratch);

  float local_sum = 0.0f;
  for (int key_patch = threadIdx.x; key_patch < rows;
       key_patch += blockDim.x) {
    const float k_variance = fmaxf(
        __half2float(qk_patch_statistics[k_statistics_offset + key_patch]),
        0.0f);
    const float bias = fminf(
        kJensenRiskCoefficient * q_energy * k_variance /
            static_cast<float>(head_dim),
        kMaximumJensenBias);
    const int count = valid_patch_tokens(
        key_patch, patch_rows, patch_columns, height, width);
    const float adjusted =
        __half2float(logits_risk[row_offset + key_patch]) +
        logf(static_cast<float>(count)) + bias;
    local_sum += expf(adjusted - row_max);
  }
  const float inverse_sum =
      1.0f / block_reduce<false>(local_sum, warp_scratch);

  for (int key_patch = threadIdx.x; key_patch < rows;
       key_patch += blockDim.x) {
    const float k_variance = fmaxf(
        __half2float(qk_patch_statistics[k_statistics_offset + key_patch]),
        0.0f);
    const float bias = fminf(
        kJensenRiskCoefficient * q_energy * k_variance /
            static_cast<float>(head_dim),
        kMaximumJensenBias);
    const int count = valid_patch_tokens(
        key_patch, patch_rows, patch_columns, height, width);
    const float adjusted =
        __half2float(logits_risk[row_offset + key_patch]) +
        logf(static_cast<float>(count)) + bias;
    const float corrected_mass = expf(adjusted - row_max) * inverse_sum;
    const float risk =
        corrected_mass * expm1f(bias) *
        fmaxf(v_sensitivity[sensitivity_offset + key_patch], 0.0f);
    logits_risk[row_offset + key_patch] =
        __float2half_rn(fminf(fmaxf(risk, 0.0f), kMaximumHalf));
  }
}

__global__ __launch_bounds__(kSoftmaxThreads)
void row_skip_error_risk_k64_kernel(
    half* __restrict__ logits_risk,
    const half* __restrict__ qk_patch_statistics,
    const half* __restrict__ k_variance64,
    const float* __restrict__ v_sensitivity,
    int64_t row_count,
    int query_rows,
    int key_groups,
    int q_heads,
    int kv_heads,
    int head_dim,
    int patch_rows,
    int patch_columns,
    int height,
    int width) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int query_patch = static_cast<int>(row % query_rows);
  const int q_head = static_cast<int>((row / query_rows) % q_heads);
  const int batch = static_cast<int>(
      row / (static_cast<int64_t>(q_heads) * query_rows));
  const int kv_head = q_head / (q_heads / kv_heads);
  const int statistics_heads = q_heads + kv_heads;
  const int64_t q_statistics_offset =
      (static_cast<int64_t>(batch) * statistics_heads + q_head) *
          query_rows +
      query_patch;
  const int64_t k_statistics_offset =
      (static_cast<int64_t>(batch) * kv_heads + kv_head) * key_groups;
  const int64_t sensitivity_offset = k_statistics_offset;
  const int64_t row_offset = row * key_groups;
  const float q_energy =
      fmaxf(__half2float(qk_patch_statistics[q_statistics_offset]), 0.0f);
  __shared__ float warp_scratch[kSoftmaxWarps];

  float local_max = -CUDART_INF_F;
  for (int key_stage = threadIdx.x; key_stage < key_groups;
       key_stage += blockDim.x) {
    const float k_variance = fmaxf(
        __half2float(k_variance64[k_statistics_offset + key_stage]), 0.0f);
    const float bias = fminf(
        kJensenRiskCoefficient * q_energy * k_variance /
            static_cast<float>(head_dim),
        kMaximumJensenBias);
    const int count = valid_k64_tokens(
        key_stage, patch_rows, patch_columns, height, width);
    const float adjusted =
        __half2float(logits_risk[row_offset + key_stage]) +
        logf(static_cast<float>(count)) + bias;
    local_max = fmaxf(local_max, adjusted);
  }
  const float row_max = block_reduce<true>(local_max, warp_scratch);

  float local_sum = 0.0f;
  for (int key_stage = threadIdx.x; key_stage < key_groups;
       key_stage += blockDim.x) {
    const float k_variance = fmaxf(
        __half2float(k_variance64[k_statistics_offset + key_stage]), 0.0f);
    const float bias = fminf(
        kJensenRiskCoefficient * q_energy * k_variance /
            static_cast<float>(head_dim),
        kMaximumJensenBias);
    const int count = valid_k64_tokens(
        key_stage, patch_rows, patch_columns, height, width);
    const float adjusted =
        __half2float(logits_risk[row_offset + key_stage]) +
        logf(static_cast<float>(count)) + bias;
    local_sum += expf(adjusted - row_max);
  }
  const float inverse_sum =
      1.0f / block_reduce<false>(local_sum, warp_scratch);

  for (int key_stage = threadIdx.x; key_stage < key_groups;
       key_stage += blockDim.x) {
    const float k_variance = fmaxf(
        __half2float(k_variance64[k_statistics_offset + key_stage]), 0.0f);
    const float bias = fminf(
        kJensenRiskCoefficient * q_energy * k_variance /
            static_cast<float>(head_dim),
        kMaximumJensenBias);
    const int count = valid_k64_tokens(
        key_stage, patch_rows, patch_columns, height, width);
    const float adjusted =
        __half2float(logits_risk[row_offset + key_stage]) +
        logf(static_cast<float>(count)) + bias;
    const float corrected_mass = expf(adjusted - row_max) * inverse_sum;
    const float risk =
        corrected_mass * expm1f(bias) *
        fmaxf(v_sensitivity[sensitivity_offset + key_stage], 0.0f);
    logits_risk[row_offset + key_stage] =
        __float2half_rn(fminf(fmaxf(risk, 0.0f), kMaximumHalf));
  }
}

void launch_draft_gemm(
    const torch::Tensor& q_pool,
    const torch::Tensor& k_pool,
    torch::Tensor& logits,
    cublasHandle_t handle,
    bool enable_released_h3_mha) {
  const int64_t batch_size = q_pool.size(0);
  const int64_t q_heads = q_pool.size(1);
  const int64_t kv_heads = k_pool.size(1);
  const int64_t query_rows = q_pool.size(2);
  const int64_t key_rows = k_pool.size(2);
  const int64_t head_dim = q_pool.size(3);
  const int64_t queries_per_kv = q_heads / kv_heads;
  const int64_t batch_kv_heads = checked_positive_product(
      batch_size, kv_heads, "Draft B*Hkv");
  const int64_t q_operand_stride = checked_positive_product(
      query_rows, head_dim, "Draft Q operand head stride");
  const int64_t k_operand_stride = checked_positive_product(
      key_rows, head_dim, "Draft K operand head stride");
  const int64_t output_stride = checked_positive_product(
      query_rows, key_rows, "Draft logits head stride");
  const int64_t grouped_q_operand_stride = checked_positive_product(
      queries_per_kv, q_operand_stride, "Draft grouped Q head stride");
  const int64_t grouped_output_stride = checked_positive_product(
      queries_per_kv, output_stride, "Draft grouped logits head stride");

  TORCH_CHECK(
      query_rows <= std::numeric_limits<int>::max() &&
          key_rows <= std::numeric_limits<int>::max(),
      "Draft row/column count exceeds cuBLAS int range");
  TORCH_CHECK(
      head_dim <= std::numeric_limits<int>::max(),
      "Draft head dimension exceeds cuBLAS int range");
  TORCH_CHECK(
      queries_per_kv <= std::numeric_limits<int>::max(),
      "Draft GQA group exceeds cuBLAS batch range");
  TORCH_CHECK(
      batch_kv_heads <= std::numeric_limits<int>::max(),
      "Draft B*Hkv exceeds cuBLAS batch range");

  const float alpha = 1.0f / std::sqrt(static_cast<float>(head_dim));
  constexpr float beta = 0.0f;
  const half* q_ptr =
      reinterpret_cast<const half*>(q_pool.data_ptr<at::Half>());
  const half* k_ptr =
      reinterpret_cast<const half*>(k_pool.data_ptr<at::Half>());
  half* logits_ptr = reinterpret_cast<half*>(logits.data_ptr<at::Half>());

  // The released H3 MHA cell owns one K head per Q head, so its contiguous
  // [B,H,R,D] operands and [B,H,R,R] output form one regular strided batch.
  // This exact cell is bitwise-equal to the established per-head calls.  Some
  // smaller shapes select a different cuBLAS batched kernel and round logits
  // differently, so keep every other MHA/GQA shape on the original path.
  const bool is_released_h3_mha =
      enable_released_h3_mha && batch_size == 1 && q_heads == 14 &&
      kv_heads == 14 && query_rows == 333 && key_rows == 333 &&
      head_dim == 128;
  if (is_released_h3_mha) {
    const int64_t mha_batches = checked_positive_product(
        batch_size, q_heads, "Draft MHA batch count");
    TORCH_CHECK(
        mha_batches <= std::numeric_limits<int>::max(),
        "Draft MHA batch count exceeds cuBLAS int range");
    check_cublas(
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(key_rows),
            static_cast<int>(query_rows),
            static_cast<int>(head_dim),
            &alpha,
            k_ptr,
            CUDA_R_16F,
            static_cast<int>(head_dim),
            k_operand_stride,
            q_ptr,
            CUDA_R_16F,
            static_cast<int>(head_dim),
            q_operand_stride,
            &beta,
            logits_ptr,
            CUDA_R_16F,
            static_cast<int>(key_rows),
            output_stride,
            static_cast<int>(mha_batches),
            kDraftComputeType,
            kDraftAlgorithm),
        "Draft MHA cublasGemmStridedBatchedEx");
    return;
  }

  // Row-major Q*K^T is emitted through the equivalent column-major K*Q^T.
  // The GQA owner map is affine along either of two axes, so choose the axis
  // requiring fewer host API calls.  Batching query heads within one owner
  // uses strideA=0.  Batching all B*Hkv owners for one fixed group offset uses
  // contiguous K heads and a Q/C stride of queries_per_kv heads.  The latter
  // turns MHA (queries_per_kv == 1) into exactly one cuBLAS call without
  // materializing a broadcast K tensor or allocating pointer arrays.
  if (queries_per_kv < batch_kv_heads) {
    for (int64_t query_in_group = 0; query_in_group < queries_per_kv;
         ++query_in_group) {
      check_cublas(
          cublasGemmStridedBatchedEx(
              handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              static_cast<int>(key_rows),
              static_cast<int>(query_rows),
              static_cast<int>(head_dim),
              &alpha,
              k_ptr,
              CUDA_R_16F,
              static_cast<int>(head_dim),
              k_operand_stride,
              q_ptr + query_in_group * q_operand_stride,
              CUDA_R_16F,
              static_cast<int>(head_dim),
              grouped_q_operand_stride,
              &beta,
              logits_ptr + query_in_group * output_stride,
              CUDA_R_16F,
              static_cast<int>(key_rows),
              grouped_output_stride,
              static_cast<int>(batch_kv_heads),
              kDraftComputeType,
              kDraftAlgorithm),
          "Draft cublasGemmStridedBatchedEx");
    }
  } else {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      for (int64_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const int64_t first_q_head = kv_head * queries_per_kv;
        const half* q_group =
            q_ptr + (batch * q_heads + first_q_head) * q_operand_stride;
        const half* k_head =
            k_ptr + (batch * kv_heads + kv_head) * k_operand_stride;
        half* logits_group =
            logits_ptr + (batch * q_heads + first_q_head) * output_stride;
        check_cublas(
            cublasGemmStridedBatchedEx(
                handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              static_cast<int>(key_rows),
              static_cast<int>(query_rows),
                static_cast<int>(head_dim),
                &alpha,
                k_head,
                CUDA_R_16F,
                static_cast<int>(head_dim),
                0,
                q_group,
                CUDA_R_16F,
                static_cast<int>(head_dim),
              q_operand_stride,
                &beta,
                logits_group,
                CUDA_R_16F,
              static_cast<int>(key_rows),
                output_stride,
                static_cast<int>(queries_per_kv),
                kDraftComputeType,
                kDraftAlgorithm),
            "Draft cublasGemmStridedBatchedEx");
      }
    }
  }
}

__global__ __launch_bounds__(kSoftmaxThreads) void blend_k64_scores_kernel(
    const half* __restrict__ k128_score,
    const half* __restrict__ k64_score,
    half* __restrict__ output,
    int64_t items_per_head,
    float alpha) {
  const int64_t head = blockIdx.x;
  const int64_t base = head * items_per_head;
  const int64_t k128_base = head * (items_per_head / 2);
  float base_sum = 0.0f;
  float fine_sum = 0.0f;
  for (int64_t item = threadIdx.x; item < items_per_head;
       item += blockDim.x) {
    base_sum += __half2float(k128_score[k128_base + (item >> 1)]);
    fine_sum += __half2float(k64_score[base + item]);
  }
  __shared__ float scratch[kSoftmaxWarps];
  base_sum = block_reduce<false>(base_sum, scratch);
  fine_sum = block_reduce<false>(fine_sum, scratch);
  const float base_scale = base_sum > 0.0f
      ? (1.0f - alpha) / base_sum
      : 0.0f;
  const float fine_scale = fine_sum > 0.0f ? alpha / fine_sum : 0.0f;
  // Routing only consumes ordering, but storing a unit-sum distribution over
  // H3's ~222k physical stages would quantize ordinary scores into FP16
  // subnormals.  Restore unit-mean magnitude after normalization so alpha has
  // a stable meaning without discarding score resolution.
  const float magnitude = static_cast<float>(items_per_head);
  for (int64_t item = threadIdx.x; item < items_per_head;
       item += blockDim.x) {
    const float blended = magnitude * (
        __half2float(k128_score[k128_base + (item >> 1)]) * base_scale +
        __half2float(k64_score[base + item]) * fine_scale);
    output[base + item] = __float2half_rn(fminf(blended, kMaximumHalf));
  }
}

}  // namespace

torch::Tensor draft_logits_impl(
    torch::Tensor q_pool,
    torch::Tensor k_pool) {
  check_pool_tensor(q_pool, "q_pool");
  check_pool_tensor(k_pool, "k_pool");
  TORCH_CHECK(
      q_pool.device() == k_pool.device(),
      "q_pool and k_pool must share one CUDA device");
  TORCH_CHECK(
      q_pool.size(0) == k_pool.size(0),
      "q_pool and k_pool batch dimensions must match");
  TORCH_CHECK(
      q_pool.size(2) == k_pool.size(2),
      "q_pool and k_pool row dimensions must match");
  TORCH_CHECK(
      q_pool.size(3) == k_pool.size(3),
      "q_pool and k_pool head dimensions must match");
  TORCH_CHECK(
      q_pool.size(1) % k_pool.size(1) == 0,
      "Q pool heads must be divisible by KV pool heads");

  c10::cuda::CUDAGuard device_guard(q_pool.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(q_pool.get_device());
  TORCH_CHECK(
      (properties->major == 8 && properties->minor == 9) ||
          (properties->major == 12 && properties->minor == 0),
      "mpa._cuda_router requires SM89 or SM120, found sm_",
      properties->major, properties->minor);

  const int64_t rows = q_pool.size(2);
  const int64_t output_head_stride = checked_positive_product(
      rows, rows, "Draft probability head stride");
  const int64_t output_heads = checked_positive_product(
      q_pool.size(0), q_pool.size(1), "Draft B*Hq");
  // Force validation of the complete output product before allocation and
  // before any cuBLAS launch.
  checked_positive_product(
      output_heads, output_head_stride, "Draft probability elements");

  auto logits = torch::empty(
      {q_pool.size(0), q_pool.size(1), rows, rows}, q_pool.options());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(q_pool.get_device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t handle_stream = nullptr;
  check_cublas(
      cublasGetStream(handle, &handle_stream), "Draft cublasGetStream");
  TORCH_CHECK(
      handle_stream == stream,
      "PyTorch cuBLAS handle is not bound to the current CUDA stream");
  cublasPointerMode_t pointer_mode;
  check_cublas(
      cublasGetPointerMode(handle, &pointer_mode),
      "Draft cublasGetPointerMode");
  TORCH_CHECK(
      pointer_mode == CUBLAS_POINTER_MODE_HOST,
      "Draft GEMM requires PyTorch's host cuBLAS pointer mode");

  launch_draft_gemm(
      q_pool,
      k_pool,
      logits,
      handle,
      properties->major == 8 && properties->minor == 9);
  return logits;
}

std::tuple<torch::Tensor, torch::Tensor> draft_probability_impl(
    torch::Tensor q_pool,
    torch::Tensor k_pool,
    bool return_row_lse) {
  auto logits = draft_logits_impl(q_pool, k_pool);
  const int64_t rows = logits.size(2);
  const int64_t output_heads = checked_positive_product(
      logits.size(0), logits.size(1), "Draft B*Hq");
  const int64_t softmax_rows = checked_positive_product(
      output_heads, rows, "Draft softmax row count");
  TORCH_CHECK(
      softmax_rows <= kMaxGridX,
      "Draft softmax grid.x exceeds CUDA limit");
  auto row_lse = return_row_lse
      ? torch::empty(
            {logits.size(0), logits.size(1), rows},
            logits.options().dtype(at::ScalarType::Float))
      : torch::empty(
            {0}, logits.options().dtype(at::ScalarType::Float));
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(logits.get_device());
  row_softmax_fp16_kernel<<<
      static_cast<unsigned int>(softmax_rows),
      kSoftmaxThreads,
      0,
      stream>>>(
      reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
      return_row_lse ? row_lse.data_ptr<float>() : nullptr,
      softmax_rows,
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {logits, row_lse};
}

torch::Tensor draft_logits(
    torch::Tensor q_pool,
    torch::Tensor k_pool) {
  return draft_logits_impl(std::move(q_pool), std::move(k_pool));
}

torch::Tensor draft_probability(
    torch::Tensor q_pool,
    torch::Tensor k_pool) {
  return std::get<0>(draft_probability_impl(q_pool, k_pool, false));
}

torch::Tensor draft_scores(
    torch::Tensor q_pool,
    torch::Tensor k_pool) {
  return draft_logits_impl(std::move(q_pool), std::move(k_pool));
}

torch::Tensor draft_skip_error_risk(
    torch::Tensor q_pool,
    torch::Tensor k_pool,
    torch::Tensor qk_patch_statistics,
    torch::Tensor v_sum,
    int64_t frames,
    int64_t height,
    int64_t width) {
  check_pool_tensor(q_pool, "q_pool");
  check_pool_tensor(k_pool, "k_pool");
  TORCH_CHECK(
      q_pool.device() == k_pool.device(),
      "q_pool and k_pool must share one CUDA device");
  TORCH_CHECK(
      q_pool.size(0) == k_pool.size(0) &&
          q_pool.size(2) == k_pool.size(2) &&
          q_pool.size(3) == k_pool.size(3),
      "q_pool and k_pool batch/row/head dimensions must match");
  TORCH_CHECK(
      q_pool.size(1) % k_pool.size(1) == 0,
      "Q pool heads must be divisible by KV pool heads");
  check_skip_risk_tensors(q_pool, k_pool, qk_patch_statistics, v_sum);
  const int64_t rows = q_pool.size(2);
  TORCH_CHECK(
      checked_patch_count(frames, height, width) == rows,
      "frames/height/width raster patch count must match R");
  const int64_t video_tokens = checked_positive_product(
      frames, checked_positive_product(height, width, "video frame tokens"),
      "video tokens");
  TORCH_CHECK(
      video_tokens <= std::numeric_limits<int>::max(),
      "video token count exceeds int32 range");
  const int64_t output_heads = checked_positive_product(
      q_pool.size(0), q_pool.size(1), "Draft-risk B*Hq");
  const int64_t softmax_rows = checked_positive_product(
      output_heads, rows, "Draft-risk softmax rows");
  const int64_t kv_owners = checked_positive_product(
      q_pool.size(0), k_pool.size(1), "Draft-risk B*Hkv");
  TORCH_CHECK(
      softmax_rows <= kMaxGridX && kv_owners * rows <= kMaxGridX,
      "Draft-risk launch grid exceeds CUDA grid.x limit");

  c10::cuda::CUDAGuard device_guard(q_pool.device());
  auto logits = torch::empty(
      {q_pool.size(0), q_pool.size(1), rows, rows}, q_pool.options());
  auto global_v_mean = torch::empty(
      {q_pool.size(0), k_pool.size(1), q_pool.size(3)},
      v_sum.options());
  auto v_sensitivity = torch::empty(
      {q_pool.size(0), k_pool.size(1), rows}, v_sum.options());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(q_pool.get_device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  launch_draft_gemm(q_pool, k_pool, logits, handle, false);

  global_v_mean_kernel<<<
      static_cast<unsigned int>(kv_owners), kSoftmaxThreads, 0, stream>>>(
      v_sum.data_ptr<float>(),
      global_v_mean.data_ptr<float>(),
      static_cast<int>(rows),
      static_cast<int>(q_pool.size(3)),
      static_cast<int>(video_tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int patch_rows = static_cast<int>((height + 7) / 8);
  const int patch_columns = static_cast<int>((width + 15) / 16);
  v_sensitivity_kernel<<<
      static_cast<unsigned int>(kv_owners * rows),
      kSoftmaxThreads,
      0,
      stream>>>(
      v_sum.data_ptr<float>(),
      global_v_mean.data_ptr<float>(),
      v_sensitivity.data_ptr<float>(),
      static_cast<int>(rows),
      static_cast<int>(q_pool.size(3)),
      patch_rows,
      patch_columns,
      static_cast<int>(height),
      static_cast<int>(width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  row_skip_error_risk_kernel<<<
      static_cast<unsigned int>(softmax_rows),
      kSoftmaxThreads,
      0,
      stream>>>(
      reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(
          qk_patch_statistics.data_ptr<at::Half>()),
      v_sensitivity.data_ptr<float>(),
      softmax_rows,
      static_cast<int>(rows),
      static_cast<int>(q_pool.size(1)),
      static_cast<int>(k_pool.size(1)),
      static_cast<int>(q_pool.size(3)),
      patch_rows,
      patch_columns,
      static_cast<int>(height),
      static_cast<int>(width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}

torch::Tensor draft_skip_error_risk_k64(
    torch::Tensor q_pool,
    torch::Tensor k_pool64,
    torch::Tensor qk_patch_statistics,
    torch::Tensor k_variance64,
    torch::Tensor v_sum64,
    int64_t frames,
    int64_t height,
    int64_t width) {
  check_pool_tensor(q_pool, "q_pool");
  check_pool_tensor(k_pool64, "k_pool64");
  TORCH_CHECK(
      q_pool.device() == k_pool64.device(),
      "q_pool and k_pool64 must share one CUDA device");
  TORCH_CHECK(
      q_pool.size(0) == k_pool64.size(0) &&
          2 * q_pool.size(2) == k_pool64.size(2) &&
          q_pool.size(3) == k_pool64.size(3),
      "k_pool64 must have twice the Q-patch rows and matching B/D");
  TORCH_CHECK(
      q_pool.size(1) % k_pool64.size(1) == 0,
      "Q pool heads must be divisible by K64 pool heads");
  const int64_t batch = q_pool.size(0);
  const int64_t q_heads = q_pool.size(1);
  const int64_t kv_heads = k_pool64.size(1);
  const int64_t query_rows = q_pool.size(2);
  const int64_t key_groups = k_pool64.size(2);
  const int64_t head_dim = q_pool.size(3);
  TORCH_CHECK(
      qk_patch_statistics.defined() && qk_patch_statistics.is_cuda() &&
          qk_patch_statistics.device() == q_pool.device() &&
          qk_patch_statistics.is_contiguous() &&
          qk_patch_statistics.scalar_type() == at::ScalarType::Half &&
          qk_patch_statistics.sizes() ==
              torch::IntArrayRef(
                  {batch, q_heads + kv_heads, query_rows}),
      "qk_patch_statistics must be FP16 [B,Hq+Hkv,R]");
  TORCH_CHECK(
      k_variance64.defined() && k_variance64.is_cuda() &&
          k_variance64.device() == q_pool.device() &&
          k_variance64.is_contiguous() &&
          k_variance64.scalar_type() == at::ScalarType::Half &&
          k_variance64.sizes() ==
              torch::IntArrayRef({batch, kv_heads, key_groups}),
      "k_variance64 must be FP16 [B,Hkv,2R]");
  TORCH_CHECK(
      v_sum64.defined() && v_sum64.is_cuda() &&
          v_sum64.device() == q_pool.device() && v_sum64.is_contiguous() &&
          v_sum64.scalar_type() == at::ScalarType::Float &&
          v_sum64.sizes() ==
              torch::IntArrayRef({batch, kv_heads, key_groups, head_dim}),
      "v_sum64 must be FP32 [B,Hkv,2R,D]");
  TORCH_CHECK(
      checked_patch_count(frames, height, width) == query_rows &&
          height % 8 == 0,
      "K64 Draft risk requires matching raster geometry with height "
      "divisible by 8");
  const int64_t video_tokens = checked_positive_product(
      frames, checked_positive_product(height, width, "video frame tokens"),
      "video tokens");
  const int64_t output_heads = checked_positive_product(
      batch, q_heads, "K64 Draft-risk B*Hq");
  const int64_t softmax_rows = checked_positive_product(
      output_heads, query_rows, "K64 Draft-risk rows");
  const int64_t kv_owners = checked_positive_product(
      batch, kv_heads, "K64 Draft-risk B*Hkv");
  TORCH_CHECK(
      softmax_rows <= kMaxGridX && kv_owners * key_groups <= kMaxGridX &&
          video_tokens <= std::numeric_limits<int>::max(),
      "K64 Draft-risk launch dimensions exceed supported ranges");

  c10::cuda::CUDAGuard device_guard(q_pool.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(q_pool.get_device());
  TORCH_CHECK(
      properties->major == 12 && properties->minor == 0,
      "K64 Draft-risk candidate requires SM120");
  auto logits = torch::empty(
      {batch, q_heads, query_rows, key_groups}, q_pool.options());
  auto global_v_mean = torch::empty(
      {batch, kv_heads, head_dim}, v_sum64.options());
  auto v_sensitivity = torch::empty(
      {batch, kv_heads, key_groups}, v_sum64.options());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(q_pool.get_device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t handle_stream = nullptr;
  check_cublas(
      cublasGetStream(handle, &handle_stream),
      "K64 Draft-risk cublasGetStream");
  TORCH_CHECK(
      handle_stream == stream,
      "PyTorch cuBLAS handle is not bound to the current CUDA stream");
  launch_draft_gemm(q_pool, k_pool64, logits, handle, false);

  global_v_mean_kernel<<<
      static_cast<unsigned int>(kv_owners), kSoftmaxThreads, 0, stream>>>(
      v_sum64.data_ptr<float>(), global_v_mean.data_ptr<float>(),
      static_cast<int>(key_groups), static_cast<int>(head_dim),
      static_cast<int>(video_tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int patch_rows = static_cast<int>((height + 7) / 8);
  const int patch_columns = static_cast<int>((width + 15) / 16);
  v_sensitivity_k64_kernel<<<
      static_cast<unsigned int>(kv_owners * key_groups),
      kSoftmaxThreads, 0, stream>>>(
      v_sum64.data_ptr<float>(), global_v_mean.data_ptr<float>(),
      v_sensitivity.data_ptr<float>(), static_cast<int>(key_groups),
      static_cast<int>(head_dim), patch_rows, patch_columns,
      static_cast<int>(height), static_cast<int>(width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  row_skip_error_risk_k64_kernel<<<
      static_cast<unsigned int>(softmax_rows), kSoftmaxThreads, 0, stream>>>(
      reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(
          qk_patch_statistics.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_variance64.data_ptr<at::Half>()),
      v_sensitivity.data_ptr<float>(), softmax_rows,
      static_cast<int>(query_rows), static_cast<int>(key_groups),
      static_cast<int>(q_heads), static_cast<int>(kv_heads),
      static_cast<int>(head_dim), patch_rows, patch_columns,
      static_cast<int>(height), static_cast<int>(width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}

torch::Tensor blend_k64_route_scores(
    torch::Tensor k128_score_fp16,
    torch::Tensor k64_score_fp16,
    double alpha) {
  TORCH_CHECK(
      k128_score_fp16.defined() && k128_score_fp16.is_cuda() &&
          k128_score_fp16.is_contiguous() &&
          k128_score_fp16.scalar_type() == at::ScalarType::Half &&
          k128_score_fp16.dim() == 4,
      "k128_score_fp16 must be contiguous CUDA FP16 [B,H,R,R]");
  TORCH_CHECK(
      k64_score_fp16.defined() && k64_score_fp16.is_cuda() &&
          k64_score_fp16.device() == k128_score_fp16.device() &&
          k64_score_fp16.is_contiguous() &&
          k64_score_fp16.scalar_type() == at::ScalarType::Half &&
          k64_score_fp16.dim() == 4,
      "k64_score_fp16 must be compatible contiguous CUDA FP16");
  const int64_t batch = k128_score_fp16.size(0);
  const int64_t heads = k128_score_fp16.size(1);
  const int64_t rows = k128_score_fp16.size(2);
  TORCH_CHECK(
      rows <= std::numeric_limits<int64_t>::max() / 2,
      "score row count exceeds int64 range");
  TORCH_CHECK(
      batch > 0 && heads > 0 && rows > 0 &&
          k128_score_fp16.size(3) == rows &&
          k64_score_fp16.sizes() ==
              torch::IntArrayRef({batch, heads, rows, 2 * rows}),
      "score tensors must have shapes [B,H,R,R] and [B,H,R,2R]");
  TORCH_CHECK(
      std::isfinite(alpha) && alpha >= 0.0 && alpha <= 1.0,
      "alpha must be finite and in [0,1]");
  const int64_t output_heads = checked_positive_product(
      batch, heads, "K64 score blend B*H");
  const int64_t items_per_head = checked_positive_product(
      rows, 2 * rows, "K64 score blend items/head");
  TORCH_CHECK(
      output_heads <= kMaxGridX,
      "K64 score blend grid exceeds CUDA grid.x limit");

  c10::cuda::CUDAGuard device_guard(k128_score_fp16.device());
  auto output = torch::empty_like(k64_score_fp16);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(k128_score_fp16.get_device());
  blend_k64_scores_kernel<<<
      static_cast<unsigned int>(output_heads), kSoftmaxThreads, 0, stream>>>(
      reinterpret_cast<const half*>(k128_score_fp16.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k64_score_fp16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      items_per_head,
      static_cast<float>(alpha));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor draft_spatial_skip_error_risk_q4_k8(
    torch::Tensor packed_q_fp16,
    torch::Tensor packed_k_fp16,
    torch::Tensor packed_v_fp16,
    torch::Tensor base_risk_fp16,
    int64_t frames,
    int64_t height,
    int64_t width,
    double residual_coefficient) {
  const auto check_packed = [&](const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.defined() && tensor.is_cuda() && tensor.is_contiguous() &&
            tensor.scalar_type() == at::ScalarType::Half && tensor.dim() == 4,
        name, " must be contiguous CUDA FP16 [B,H,R*128,D128]");
    TORCH_CHECK(
        tensor.size(0) > 0 && tensor.size(1) > 0 && tensor.size(2) > 0 &&
            tensor.size(3) == kSpatialHeadDim,
        name, " must be nonempty with head dimension 128");
  };
  check_packed(packed_q_fp16, "packed_q_fp16");
  check_packed(packed_k_fp16, "packed_k_fp16");
  check_packed(packed_v_fp16, "packed_v_fp16");
  TORCH_CHECK(
      packed_q_fp16.device() == packed_k_fp16.device() &&
          packed_q_fp16.device() == packed_v_fp16.device(),
      "packed Q/K/V must share one CUDA device");
  TORCH_CHECK(
      packed_k_fp16.sizes() == packed_v_fp16.sizes(),
      "packed K and V shapes must match");
  TORCH_CHECK(
      packed_q_fp16.size(0) == packed_k_fp16.size(0) &&
          packed_q_fp16.size(2) == packed_k_fp16.size(2) &&
          packed_q_fp16.size(1) % packed_k_fp16.size(1) == 0,
      "packed Q must match K/V batch/token dimensions and use divisible GQA");
  const int64_t patches = checked_patch_count(frames, height, width);
  TORCH_CHECK(
      packed_q_fp16.size(2) ==
          checked_positive_product(
              patches, kRasterPatchTokens, "spatial-risk packed tokens"),
      "packed token length must equal raster patch count * 128");
  const int64_t batch = packed_q_fp16.size(0);
  const int64_t q_heads = packed_q_fp16.size(1);
  const int64_t kv_heads = packed_k_fp16.size(1);
  TORCH_CHECK(
      base_risk_fp16.defined() && base_risk_fp16.is_cuda() &&
          base_risk_fp16.device() == packed_q_fp16.device() &&
          base_risk_fp16.is_contiguous() &&
          base_risk_fp16.scalar_type() == at::ScalarType::Half &&
          base_risk_fp16.sizes() ==
              torch::IntArrayRef({batch, q_heads, patches, patches}),
      "base_risk_fp16 must be compatible contiguous CUDA FP16 [B,Hq,R,R]");
  TORCH_CHECK(
      std::isfinite(residual_coefficient) && residual_coefficient >= 0.0 &&
          residual_coefficient <= 1024.0,
      "residual_coefficient must be finite and in [0,1024]");
  const int64_t q_owner_patches = checked_positive_product(
      checked_positive_product(batch, q_heads, "spatial-risk B*Hq"),
      patches,
      "spatial-risk Q owner patches");
  const int64_t kv_owner_patches = checked_positive_product(
      checked_positive_product(batch, kv_heads, "spatial-risk B*Hkv"),
      patches,
      "spatial-risk KV owner patches");
  const int64_t fine_query_rows = checked_positive_product(
      q_owner_patches, kSpatialQueryGroups, "spatial-risk fine query rows");
  const int64_t fine_key_rows = checked_positive_product(
      patches, kSpatialKeyGroups, "spatial-risk fine key rows");
  TORCH_CHECK(
      q_owner_patches <= kMaxGridX && kv_owner_patches <= kMaxGridX &&
          fine_query_rows <= kMaxGridX &&
          fine_query_rows <= std::numeric_limits<int64_t>::max() /
              fine_key_rows,
      "spatial-risk launch or probability shape exceeds supported range");

  c10::cuda::CUDAGuard device_guard(packed_q_fp16.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(packed_q_fp16.get_device());
  TORCH_CHECK(
      properties->major == 12 && properties->minor == 0,
      "Q4xK8 spatial error-risk candidate requires SM120, found sm_",
      properties->major,
      properties->minor);
  auto q_centroid = torch::empty(
      {batch, q_heads, patches * kSpatialQueryGroups, kSpatialHeadDim},
      packed_q_fp16.options());
  auto k_centroid = torch::empty(
      {batch, kv_heads, patches * kSpatialKeyGroups, kSpatialHeadDim},
      packed_k_fp16.options());
  auto centered_v_sketch = torch::empty(
      {batch, kv_heads, patches * kSpatialKeyGroups, kSpatialValueSketch},
      packed_v_fp16.options());
  auto fine_probability = torch::empty(
      {batch,
       q_heads,
       patches * kSpatialQueryGroups,
       patches * kSpatialKeyGroups},
      packed_q_fp16.options());
  auto output = torch::empty_like(base_risk_fp16);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(packed_q_fp16.get_device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t handle_stream = nullptr;
  check_cublas(
      cublasGetStream(handle, &handle_stream),
      "Q4xK8 spatial-risk cublasGetStream");
  TORCH_CHECK(
      handle_stream == stream,
      "PyTorch cuBLAS handle is not bound to the current CUDA stream");
  const int patch_rows = static_cast<int>((height + 7) / 8);
  const int patch_columns = static_cast<int>((width + 15) / 16);
  q4_spatial_centroid_kernel<<<
      static_cast<unsigned int>(q_owner_patches),
      kSpatialHeadDim,
      0,
      stream>>>(
      reinterpret_cast<const half*>(
          packed_q_fp16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(q_centroid.data_ptr<at::Half>()),
      static_cast<int>(patches),
      patch_rows,
      patch_columns,
      static_cast<int>(height),
      static_cast<int>(width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  k8_centroid_centered_vsketch_kernel<<<
      static_cast<unsigned int>(kv_owner_patches),
      kSpatialHeadDim,
      0,
      stream>>>(
      reinterpret_cast<const half*>(
          packed_k_fp16.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(
          packed_v_fp16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(k_centroid.data_ptr<at::Half>()),
      reinterpret_cast<half*>(centered_v_sketch.data_ptr<at::Half>()),
      static_cast<int>(patches),
      patch_rows,
      patch_columns,
      static_cast<int>(height),
      static_cast<int>(width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch_draft_gemm(
      q_centroid, k_centroid, fine_probability, handle, false);
  row_spatial_probability_fp16_kernel<<<
      static_cast<unsigned int>(fine_query_rows),
      kSoftmaxThreads,
      0,
      stream>>>(
      reinterpret_cast<half*>(fine_probability.data_ptr<at::Half>()),
      fine_query_rows,
      static_cast<int>(fine_key_rows),
      static_cast<int>(patches),
      patch_rows,
      patch_columns,
      static_cast<int>(height),
      static_cast<int>(width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  spatial_residual_score_kernel<<<
      static_cast<unsigned int>(q_owner_patches),
      kSoftmaxThreads,
      0,
      stream>>>(
      reinterpret_cast<const half*>(
          fine_probability.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(
          centered_v_sketch.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(base_risk_fp16.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      static_cast<int>(patches),
      static_cast<int>(q_heads),
      static_cast<int>(kv_heads),
      patch_rows,
      patch_columns,
      static_cast<int>(height),
      static_cast<int>(width),
      static_cast<float>(residual_coefficient));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> draft_probability_with_lse(
    torch::Tensor q_pool,
    torch::Tensor k_pool) {
  return draft_probability_impl(q_pool, k_pool, true);
}
