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
 * Project-owned exact global routing boundary. Stored FP16 probabilities are
 * stably ranked per (batch, query-head) by CUB, classified with exact host
 * counts, and compacted into one phase-packed logical-ID table.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "../api.h"

namespace {

constexpr int kThreads = 256;
constexpr int kFullFp16KeyBits = 16;
constexpr int kCompositeSegmentBits = 16;
constexpr int kMaxCompositeSegments = 1 << kCompositeSegmentBits;
constexpr int64_t kMaxGridX = 2147483647LL;
constexpr uint8_t kSkip = 0;
constexpr uint8_t kInt4 = 1;
constexpr uint8_t kInt8 = 2;
constexpr uint8_t kFp16 = 3;

struct PhaseCounts {
  int fp8;
  int fp16;
  int fp4;
};

struct AddPhaseCounts {
  __device__ __forceinline__ PhaseCounts operator()(
      const PhaseCounts& lhs, const PhaseCounts& rhs) const {
    return {
        lhs.fp8 + rhs.fp8,
        lhs.fp16 + rhs.fp16,
        lhs.fp4 + rhs.fp4};
  }
};

using PhaseRowReduce = cub::BlockReduce<PhaseCounts, kThreads>;
using PhaseRowScan = cub::BlockScan<PhaseCounts, kThreads>;
using ValidationIntegerReduce = cub::BlockReduce<int, kThreads>;
using ValidationSum = cub::BlockReduce<int64_t, kThreads>;
using SolFloatReduce = cub::BlockReduce<float, kThreads>;
using SolIntegerReduce = cub::BlockReduce<int, kThreads>;

union RowTempStorage {
  typename PhaseRowReduce::TempStorage reduce;
  typename PhaseRowScan::TempStorage scan;
};

union ValidationTempStorage {
  typename ValidationIntegerReduce::TempStorage integer_reduce;
  typename ValidationSum::TempStorage sum_reduce;
};

inline int64_t checked_positive_product(
    int64_t lhs, int64_t rhs, const char* description) {
  TORCH_CHECK(lhs > 0 && rhs > 0, description, " factors must be positive");
  TORCH_CHECK(
      lhs <= std::numeric_limits<int64_t>::max() / rhs,
      description, " exceeds int64 range");
  return lhs * rhs;
}

inline int require_nonnegative_int32(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0, name, " must be nonnegative");
  TORCH_CHECK(
      value <= std::numeric_limits<int>::max(),
      name, " exceeds int32 range");
  return static_cast<int>(value);
}

inline int required_unsigned_bits(int value_count) {
  TORCH_CHECK(value_count > 0, "value_count must be positive");
  int bits = 0;
  int values_minus_one = value_count - 1;
  while (values_minus_one > 0) {
    ++bits;
    values_minus_one >>= 1;
  }
  return bits;
}

inline void check_probability(const torch::Tensor& probability) {
  TORCH_CHECK(probability.defined(), "probability_fp16 must be defined");
  TORCH_CHECK(
      probability.is_cuda(), "probability_fp16 must be a CUDA tensor");
  TORCH_CHECK(
      probability.is_contiguous(),
      "probability_fp16 must be contiguous BHRR");
  TORCH_CHECK(
      probability.scalar_type() == at::ScalarType::Half,
      "probability_fp16 must have dtype torch.float16");
  TORCH_CHECK(
      probability.dim() == 4,
      "probability_fp16 must have shape [B,H,Q,K]");
  TORCH_CHECK(
      probability.size(0) > 0 && probability.size(1) > 0 &&
          probability.size(2) > 0 && probability.size(3) > 0,
      "probability_fp16 dimensions must be positive");
  // This is the trusted boundary after the stored-FP16 softmax. A full device scan
  // for finite/nonnegative values would duplicate work in the timed path.
}

inline void check_sol_pool_tensor(
    const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous [B,H,R,D]");
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Half,
      name, " must have dtype torch.float16");
  TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B,H,R,D]");
  TORCH_CHECK(
      tensor.size(0) > 0 && tensor.size(1) > 0 && tensor.size(2) > 0,
      name, " dimensions must be positive");
  TORCH_CHECK(
      tensor.size(3) == 64 || tensor.size(3) == 128,
      name, " head dimension must be 64 or 128");
}

inline void check_count_tensor(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Int,
      name, " must have dtype torch.int32");
  TORCH_CHECK(tensor.dim() == 3, name, " must have shape [B,H,R]");
  TORCH_CHECK(
      tensor.sizes() == reference.sizes(),
      name, " shape must match fp8_counts");
  TORCH_CHECK(
      tensor.device() == reference.device(),
      name, " device must match fp8_counts");
}

__global__ void initialize_sort_metadata_kernel(
    int* __restrict__ initial_ids,
    int* __restrict__ segment_offsets,
    int total_items,
    int segments,
    int segment_items) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total_items) {
    initial_ids[index] = static_cast<int>(index) % segment_items;
  }
  if (index <= segments) {
    segment_offsets[index] = static_cast<int>(index) * segment_items;
  }
}

__global__ void initialize_composite_sort_kernel(
    const half* __restrict__ probability,
    uint32_t* __restrict__ composite_keys,
    int* __restrict__ initial_ids,
    int total_items,
    int segment_items,
    int columns,
    int blocks_per_frame,
    int patches_w) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_items) {
    return;
  }
  const int segment = static_cast<int>(index) / segment_items;
  const int flat_id = static_cast<int>(index) - segment * segment_items;
  bool spatial_cross_anchor = false;
  if (blocks_per_frame > 0) {
    const int query = flat_id / columns;
    const int key = flat_id - query * columns;
    const int query_frame = query / blocks_per_frame;
    const int key_frame = key / blocks_per_frame;
    const int query_spatial = query - query_frame * blocks_per_frame;
    const int key_spatial = key - key_frame * blocks_per_frame;
    const int query_row = query_spatial / patches_w;
    const int key_row = key_spatial / patches_w;
    const int query_column = query_spatial - query_row * patches_w;
    const int key_column = key_spatial - key_row * patches_w;
    spatial_cross_anchor = query_frame == key_frame &&
        abs(query_row - key_row) + abs(query_column - key_column) <= 1;
  }
  // The router stores finite nonnegative FP16 probabilities. Their sign-cleared
  // binary16 representation is monotonic in value. Clearing the sign also
  // makes -0 and +0 one stable tie, matching numeric CUB half comparison.
  const uint16_t probability_bits =
      __half_as_ushort(probability[index]) & uint16_t{0x7fff};
  // All finite nonnegative FP16 probabilities have descending keys well above
  // zero.  Reserving zero for anchors makes the stable global radix sort place
  // them first without changing the relative order of any non-anchor score.
  const uint32_t descending_probability = spatial_cross_anchor
      ? uint32_t{0}
      : uint32_t{0xffff} - static_cast<uint32_t>(probability_bits);
  composite_keys[index] =
      (static_cast<uint32_t>(segment) << kFullFp16KeyBits) |
      descending_probability;
  initial_ids[index] = flat_id;
}

__global__ void scatter_precision_kernel(
    const int* __restrict__ sorted_ids,
    uint8_t* __restrict__ precision_map,
    const int* __restrict__ fp16_blocks_by_head,
    int total_items,
    int segment_items,
    int n16_end,
    int n8_end,
    int keep) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_items) {
    return;
  }
  const int rank = static_cast<int>(linear) % segment_items;
  const int segment = static_cast<int>(linear) / segment_items;
  const int segment_n16_end = fp16_blocks_by_head == nullptr
      ? n16_end
      : max(0, min(keep, fp16_blocks_by_head[segment]));
  const int segment_n8_end = fp16_blocks_by_head == nullptr ? n8_end : keep;
  uint8_t code = kSkip;
  if (rank < segment_n16_end) {
    code = kFp16;
  } else if (rank < segment_n8_end) {
    code = kInt8;
  } else if (rank < keep) {
    code = kInt4;
  }
  const int flat_id = sorted_ids[linear];
  precision_map[segment * segment_items + flat_id] = code;
}

__global__ __launch_bounds__(kThreads) void normalize_sol_rows_kernel(
    const half* __restrict__ scores,
    float* __restrict__ normalized_scores,
    int* __restrict__ selected_per_segment,
    int row_count,
    int columns,
    float beta,
    bool force_dense) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int row_offset = row * columns;
  __shared__ typename SolFloatReduce::TempStorage float_temp;
  __shared__ typename SolIntegerReduce::TempStorage integer_temp;
  __shared__ float row_mean;
  __shared__ float row_std;

  float local_sum = 0.0f;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    local_sum += __half2float(scores[row_offset + column]);
  }
  const float sum = SolFloatReduce(float_temp).Sum(local_sum);
  if (threadIdx.x == 0) {
    row_mean = sum / static_cast<float>(columns);
  }
  __syncthreads();

  float local_variance = 0.0f;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float delta = __half2float(scores[row_offset + column]) - row_mean;
    local_variance += delta * delta;
  }
  const float variance_sum = SolFloatReduce(float_temp).Sum(local_variance);
  if (threadIdx.x == 0) {
    // Match the official Sol reference: population variance and epsilon
    // inside sqrt before applying the standardized cutoff.
    row_std = sqrtf(variance_sum / static_cast<float>(columns) + 1.0e-6f);
  }
  __syncthreads();

  int local_selected = 0;
  const float threshold = row_mean + beta * row_std;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float score = __half2float(scores[row_offset + column]);
    const bool selected = force_dense || score > threshold;
    normalized_scores[row_offset + column] = selected
        ? (score - row_mean) / row_std
        : -INFINITY;
    local_selected += selected;
  }
  const int selected = SolIntegerReduce(integer_temp).Sum(local_selected);
  if (threadIdx.x == 0) {
    atomicAdd(selected_per_segment + row / columns, selected);
  }
}

__global__ void allocate_sol_phase_totals_kernel(
    const int* __restrict__ selected_per_segment,
    int* __restrict__ phase_totals,
    int segments,
    double ratio16,
    double ratio8,
    double ratio4) {
  const int segment = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (segment >= segments) {
    return;
  }
  const int keep = selected_per_segment[segment];
  const double quota16 = keep * ratio16;
  const double quota8 = keep * ratio8;
  const double quota4 = keep * ratio4;
  int count16 = static_cast<int>(floor(quota16));
  int count8 = static_cast<int>(floor(quota8));
  int count4 = static_cast<int>(floor(quota4));
  double fraction16 = quota16 - count16;
  double fraction8 = quota8 - count8;
  double fraction4 = quota4 - count4;
  const int remaining = keep - count16 - count8 - count4;
  for (int seat = 0; seat < remaining; ++seat) {
    // The >= comparisons encode the tie priority FP16 > INT8 > INT4.
    if (fraction16 >= fraction8 && fraction16 >= fraction4) {
      ++count16;
      fraction16 = -1.0;
    } else if (fraction8 >= fraction4) {
      ++count8;
      fraction8 = -1.0;
    } else {
      ++count4;
      fraction4 = -1.0;
    }
  }
  phase_totals[segment * 3 + 0] = count16;
  phase_totals[segment * 3 + 1] = count8;
  phase_totals[segment * 3 + 2] = count4;
}

__global__ void scatter_sol_precision_kernel(
    const int* __restrict__ sorted_ids,
    const int* __restrict__ selected_per_segment,
    const int* __restrict__ phase_totals,
    uint8_t* __restrict__ precision_map,
    int total_items,
    int segment_items) {
  const int linear = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_items) {
    return;
  }
  const int segment = linear / segment_items;
  const int rank = linear - segment * segment_items;
  const int n16 = phase_totals[segment * 3 + 0];
  const int n8 = phase_totals[segment * 3 + 1];
  const int keep = selected_per_segment[segment];
  uint8_t code = kSkip;
  if (rank < n16) {
    code = kFp16;
  } else if (rank < n16 + n8) {
    code = kInt8;
  } else if (rank < keep) {
    code = kInt4;
  }
  precision_map[segment * segment_items + sorted_ids[linear]] = code;
}

__global__ __launch_bounds__(kThreads) void pack_precision_rows_kernel(
    const uint8_t* __restrict__ precision_map,
    int* __restrict__ packed_ids,
    int* __restrict__ fp8_counts,
    int* __restrict__ fp16_counts,
    int* __restrict__ fp4_counts,
    int* __restrict__ first_empty_row,
    int row_count,
    int columns) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int row_offset = row * columns;
  __shared__ RowTempStorage temp;
  __shared__ int totals[3];
  __shared__ int running[3];

  PhaseCounts local_counts{0, 0, 0};
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const uint8_t code = precision_map[row_offset + column];
    local_counts.fp8 += code == kInt8;
    local_counts.fp16 += code == kFp16;
    local_counts.fp4 += code == kInt4;
    // Every output slot is initialized in the count pass. The compact pass
    // overwrites exactly the retained prefix, leaving an explicit zero tail.
    packed_ids[row_offset + column] = 0;
  }

  // CUB accepts trivially-copyable aggregate values with an associative custom
  // operation.  Reducing all three phase counters together avoids repeating
  // the block-wide exchange and synchronization for each precision label.
  const PhaseCounts row_totals =
      PhaseRowReduce(temp.reduce).Reduce(local_counts, AddPhaseCounts{});
  __syncthreads();
  if (threadIdx.x == 0) {
    totals[0] = row_totals.fp8;
    totals[1] = row_totals.fp16;
    totals[2] = row_totals.fp4;
    running[0] = 0;
    running[1] = 0;
    running[2] = 0;
    fp8_counts[row] = row_totals.fp8;
    fp16_counts[row] = row_totals.fp16;
    fp4_counts[row] = row_totals.fp4;
    if (row_totals.fp8 + row_totals.fp16 + row_totals.fp4 == 0) {
      // first_empty_row starts at int32(-1), whose unsigned representation is
      // UINT_MAX. unsigned atomicMin therefore preserves -1 for no error and
      // deterministically records the smallest linear [B,H,R] row otherwise.
      atomicMin(
          reinterpret_cast<unsigned int*>(first_empty_row),
          static_cast<unsigned int>(row));
    }
  }
  __syncthreads();

  for (int tile = 0; tile < columns; tile += kThreads) {
    const int column = tile + threadIdx.x;
    const uint8_t code =
        column < columns ? precision_map[row_offset + column] : kSkip;
    const bool valid = column < columns;
    const PhaseCounts is_phase{
        valid && code == kInt8,
        valid && code == kFp16,
        valid && code == kInt4};
    PhaseCounts prefix;
    PhaseCounts tile_counts;

    PhaseRowScan(temp.scan).ExclusiveScan(
        is_phase,
        prefix,
        PhaseCounts{0, 0, 0},
        AddPhaseCounts{},
        tile_counts);
    __syncthreads();

    // Execution order is INT4 -> INT8 -> FP16.  Count tensor return order is
    // intentionally unchanged for ABI compatibility.
    const int fp4_base = running[2];
    const int fp8_base = totals[2] + running[0];
    const int fp16_base = totals[2] + totals[0] + running[1];
    if (is_phase.fp8) {
      packed_ids[row_offset + fp8_base + prefix.fp8] = column;
    } else if (is_phase.fp16) {
      packed_ids[row_offset + fp16_base + prefix.fp16] = column;
    } else if (is_phase.fp4) {
      packed_ids[row_offset + fp4_base + prefix.fp4] = column;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      running[0] += tile_counts.fp8;
      running[1] += tile_counts.fp16;
      running[2] += tile_counts.fp4;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ int raster_patch_token_count(
    int patch,
    int height,
    int width,
    int pool_width) {
  const int patches_h = (height + 7) / 8;
  const int patches_w = (width + pool_width - 1) / pool_width;
  const int patch_in_frame = patch % (patches_h * patches_w);
  const int patch_y = patch_in_frame / patches_w;
  const int patch_x = patch_in_frame - patch_y * patches_w;
  const int valid_rows = min(8, height - patch_y * 8);
  const int valid_columns = min(pool_width, width - patch_x * pool_width);
  return valid_rows * valid_columns;
}

__device__ __forceinline__ float raster_patch_log2_mass(
    int patch,
    int height,
    int width,
    int pool_width,
    bool partial_logmass) {
  if (!partial_logmass) {
    return 0.0f;
  }
  const int count = raster_patch_token_count(
      patch, height, width, pool_width);
  return log2f(
      static_cast<float>(count) /
      static_cast<float>(8 * pool_width));
}

// Sol-Attention's default "diag" threshold uses the mean and diagonal
// variance of K centroids across key blocks.  This producer is shared across
// all query rows and heads owned by the same KV head.  k_pool contains one
// FP16 mean per logical raster patch (Q64 or Q128).  The optional prefix operand
// is the already-compacted FP16 K used by the exact dense-text phase.  It is
// reduced into independent contiguous centroids matching the video route token
// count (K64 for 8x8, K128 for 8x16) only for these moments; prefix keys never
// enter the routed video map or cease to execute exactly.
__global__ __launch_bounds__(kThreads) void sol_k_diag_stats_kernel(
    const half* __restrict__ k_pool,
    const half* __restrict__ prefix_k,
    const int32_t* __restrict__ prefix_counts,
    float* __restrict__ k_mean,
    float* __restrict__ k_variance,
    float* __restrict__ k_logmass_covariance,
    float* __restrict__ logmass_mean,
    float* __restrict__ logmass_variance,
    int owners,
    int kv_heads,
    int video_rows,
    int prefix_capacity,
    int head_dim,
    int height,
    int width,
    int pool_width,
    bool partial_logmass) {
  const int owner = static_cast<int>(blockIdx.x);
  if (owner >= owners) {
    return;
  }
  const int64_t pool_base =
      static_cast<int64_t>(owner) * video_rows * head_dim;
  const int64_t stats_base = static_cast<int64_t>(owner) * head_dim;
  const int batch = owner / kv_heads;
  const int32_t prefix_tokens = prefix_counts == nullptr
      ? 0
      : max(0, min(prefix_counts[batch], prefix_capacity));
  const int prefix_block_tokens = 8 * pool_width;
  const int prefix_blocks =
      (prefix_tokens + prefix_block_tokens - 1) / prefix_block_tokens;
  const int statistic_rows = video_rows + prefix_blocks;
  const float inverse_rows = 1.0f / static_cast<float>(statistic_rows);
  for (int dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    float sum_value_logmass = 0.0f;
    float sum_logmass = 0.0f;
    float sum_logmass_sq = 0.0f;
    for (int key = 0; key < video_rows; ++key) {
      const float value = __half2float(
          k_pool[pool_base + static_cast<int64_t>(key) * head_dim + dimension]);
      const float key_logmass = raster_patch_log2_mass(
          key, height, width, pool_width, partial_logmass);
      sum += value;
      sum_sq += value * value;
      sum_value_logmass += value * key_logmass;
      sum_logmass += key_logmass;
      sum_logmass_sq += key_logmass * key_logmass;
    }
    if (prefix_k != nullptr) {
      const int64_t prefix_base =
          static_cast<int64_t>(owner) * prefix_capacity * head_dim;
      for (int block = 0; block < prefix_blocks; ++block) {
        const int begin = block * prefix_block_tokens;
        const int end = min(begin + prefix_block_tokens, prefix_tokens);
        float block_sum = 0.0f;
        for (int token = begin; token < end; ++token) {
          block_sum += __half2float(prefix_k[
              prefix_base + static_cast<int64_t>(token) * head_dim + dimension]);
        }
        const float centroid = block_sum / static_cast<float>(end - begin);
        sum += centroid;
        sum_sq += centroid * centroid;
      }
    }
    const float mean = sum * inverse_rows;
    k_mean[stats_base + dimension] = mean;
    k_variance[stats_base + dimension] =
        fmaxf(sum_sq * inverse_rows - mean * mean, 0.0f);
    if (k_logmass_covariance != nullptr) {
      const float delta_mean = sum_logmass * inverse_rows;
      k_logmass_covariance[stats_base + dimension] =
          sum_value_logmass * inverse_rows - mean * delta_mean;
      if (dimension == 0) {
        logmass_mean[owner] = delta_mean;
        logmass_variance[owner] = fmaxf(
            sum_logmass_sq * inverse_rows - delta_mean * delta_mean,
            0.0f);
      }
    }
  }
}

template <bool ExactThreshold>
__global__ __launch_bounds__(kThreads) void sol_threshold_route_kernel(
    const half* __restrict__ logits,
    const half* __restrict__ q_pool,
    const float* __restrict__ k_mean,
    const float* __restrict__ k_variance,
    const float* __restrict__ k_logmass_covariance,
    const float* __restrict__ logmass_mean,
    const float* __restrict__ logmass_variance,
    uint8_t* __restrict__ precision_map,
    int row_count,
    int q_heads,
    int kv_heads,
    int rows,
    int head_dim,
    float low8_tau,
    float fp16_tau,
    int local_fp16_radius,
    int forced_sink_blocks,
    int height,
    int width,
    int pool_width,
    bool partial_logmass) {
  const int linear_row = static_cast<int>(blockIdx.x);
  if (linear_row >= row_count) {
    return;
  }
  const int query = linear_row % rows;
  const int query_head_owner = linear_row / rows;
  const int query_head = query_head_owner % q_heads;
  const int batch = query_head_owner / q_heads;
  const int queries_per_kv = q_heads / kv_heads;
  const int kv_head = query_head / queries_per_kv;
  const int64_t logits_base = static_cast<int64_t>(linear_row) * rows;
  constexpr float kLog2E = 1.4426950408889634f;

  __shared__ typename SolFloatReduce::TempStorage reduction;
  __shared__ float row_mean;
  __shared__ float row_std;

  if constexpr (ExactThreshold) {
    float local_sum = 0.0f;
    for (int key = threadIdx.x; key < rows; key += blockDim.x) {
      local_sum +=
          __half2float(logits[logits_base + key]) * kLog2E +
          raster_patch_log2_mass(
              key, height, width, pool_width, partial_logmass);
    }
    const float sum = SolFloatReduce(reduction).Sum(local_sum);
    __syncthreads();
    if (threadIdx.x == 0) {
      row_mean = sum / static_cast<float>(rows);
    }
    __syncthreads();
    float local_squared_deviation = 0.0f;
    for (int key = threadIdx.x; key < rows; key += blockDim.x) {
      const float value =
          __half2float(logits[logits_base + key]) * kLog2E +
          raster_patch_log2_mass(
              key, height, width, pool_width, partial_logmass) -
          row_mean;
      local_squared_deviation += value * value;
    }
    const float squared_deviation =
        SolFloatReduce(reduction).Sum(local_squared_deviation);
    __syncthreads();
    if (threadIdx.x == 0) {
      row_std = sqrtf(
          fmaxf(squared_deviation / static_cast<float>(rows), 0.0f) + 1.0e-6f);
    }
  } else {
    const int64_t q_base =
        (static_cast<int64_t>(batch) * q_heads + query_head) * rows * head_dim +
        static_cast<int64_t>(query) * head_dim;
    const int64_t stats_base =
        (static_cast<int64_t>(batch) * kv_heads + kv_head) * head_dim;
    float local_mean = 0.0f;
    float local_variance = 0.0f;
    float local_logmass_covariance = 0.0f;
    for (int dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
      const float q = __half2float(q_pool[q_base + dimension]);
      local_mean += q * k_mean[stats_base + dimension];
      local_variance += q * q * k_variance[stats_base + dimension];
      if (k_logmass_covariance != nullptr) {
        local_logmass_covariance +=
            q * k_logmass_covariance[stats_base + dimension];
      }
    }
    const float raw_mean = SolFloatReduce(reduction).Sum(local_mean);
    __syncthreads();
    const float raw_variance = SolFloatReduce(reduction).Sum(local_variance);
    __syncthreads();
    const float raw_logmass_covariance =
        SolFloatReduce(reduction).Sum(local_logmass_covariance);
    __syncthreads();
    if (threadIdx.x == 0) {
      const float log2_scale =
          kLog2E / sqrtf(static_cast<float>(head_dim));
      const float delta_mean = logmass_mean == nullptr
          ? 0.0f
          : logmass_mean[batch * kv_heads + kv_head];
      const float delta_variance = logmass_variance == nullptr
          ? 0.0f
          : logmass_variance[batch * kv_heads + kv_head];
      row_mean = raw_mean * log2_scale + delta_mean;
      row_std = sqrtf(
          fmaxf(
              raw_variance * log2_scale * log2_scale +
                  delta_variance +
                  2.0f * raw_logmass_covariance * log2_scale,
              0.0f) +
          1.0e-6f);
    }
  }
  __syncthreads();

  const float low8_threshold = row_mean + low8_tau * row_std;
  const float fp16_threshold = row_mean + fp16_tau * row_std;
  for (int key = threadIdx.x; key < rows; key += blockDim.x) {
    const int distance = query > key ? query - key : key - query;
    uint8_t code = kSkip;
    if (key < forced_sink_blocks || distance <= local_fp16_radius) {
      code = kFp16;
    } else {
      const float score =
          __half2float(logits[logits_base + key]) * kLog2E +
          raster_patch_log2_mass(
              key, height, width, pool_width, partial_logmass);
      // Match Sol's strict comparison.  The higher threshold selects the
      // expensive rescue phase; the original tau boundary selects the union
      // of FP8 and FP16 exact-compute blocks.
      if (score > fp16_threshold) {
        code = kFp16;
      } else if (score > low8_threshold) {
        code = kInt8;
      }
    }
    precision_map[logits_base + key] = code;
  }
}

__global__ __launch_bounds__(kThreads) void validate_route_metadata_kernel(
    const int* __restrict__ fp8_counts,
    const int* __restrict__ fp16_counts,
    const int* __restrict__ fp4_counts,
    const int* __restrict__ first_empty_row,
    int* __restrict__ validation,
    int segments,
    int rows,
    int n8,
    int n16,
    int n4) {
  __shared__ ValidationTempStorage temp;
  __shared__ int all_counts_valid;
  __shared__ int expected_first_empty;
  if (threadIdx.x == 0) {
    all_counts_valid = 1;
    expected_first_empty = segments * rows;
  }
  __syncthreads();

  for (int segment = 0; segment < segments; ++segment) {
    int64_t local8 = 0;
    int64_t local16 = 0;
    int64_t local4 = 0;
    int local_valid = 1;
    int local_first = segments * rows;
    for (int query = threadIdx.x; query < rows; query += blockDim.x) {
      const int index = segment * rows + query;
      const int count8 = fp8_counts[index];
      const int count16 = fp16_counts[index];
      const int count4 = fp4_counts[index];
      const int64_t row_total =
          static_cast<int64_t>(count8) + count16 + count4;
      local8 += count8;
      local16 += count16;
      local4 += count4;
      local_valid &= count8 >= 0 && count8 <= rows &&
          count16 >= 0 && count16 <= rows &&
          count4 >= 0 && count4 <= rows &&
          row_total <= rows;
      if (row_total == 0) {
        local_first = min(local_first, index);
      }
    }

    const int64_t sum8 = ValidationSum(temp.sum_reduce).Sum(local8);
    __syncthreads();
    const int64_t sum16 = ValidationSum(temp.sum_reduce).Sum(local16);
    __syncthreads();
    const int64_t sum4 = ValidationSum(temp.sum_reduce).Sum(local4);
    __syncthreads();
    const int segment_valid =
        ValidationIntegerReduce(temp.integer_reduce).Reduce(
            local_valid, cub::Min());
    __syncthreads();
    const int segment_first =
        ValidationIntegerReduce(temp.integer_reduce).Reduce(
            local_first, cub::Min());
    __syncthreads();
    if (threadIdx.x == 0) {
      all_counts_valid &= segment_valid && sum8 == n8 && sum16 == n16 &&
          sum4 == n4;
      expected_first_empty = min(expected_first_empty, segment_first);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (expected_first_empty == segments * rows) {
      expected_first_empty = -1;
    }
    validation[0] = all_counts_valid;
    validation[1] = first_empty_row[0] == expected_first_empty;
    validation[2] = expected_first_empty;
  }
}

}  // namespace

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision_impl(
    torch::Tensor probability_fp16,
    int64_t n16_value,
    int64_t n8_value,
    int64_t n4_value,
    bool request_global_sort,
    const torch::Tensor* fp16_blocks_by_head = nullptr,
    int spatial_cross_blocks_per_frame = 0,
    int spatial_cross_patches_w = 0) {
  check_probability(probability_fp16);
  const int n16 = require_nonnegative_int32(n16_value, "n16");
  const int n8 = require_nonnegative_int32(n8_value, "n8");
  const int n4 = require_nonnegative_int32(n4_value, "n4");
  TORCH_CHECK(
      n16_value <= std::numeric_limits<int64_t>::max() - n8_value &&
          n16_value + n8_value <=
              std::numeric_limits<int64_t>::max() - n4_value,
      "n16+n8+n4 exceeds int64 range");
  const int64_t keep_value = n16_value + n8_value + n4_value;
  TORCH_CHECK(keep_value > 0, "retained count must be positive");

  const int64_t rows = probability_fp16.size(2);
  const int64_t columns = probability_fp16.size(3);
  const int64_t segment_items_value = checked_positive_product(
      rows, columns, "sort items per head");
  TORCH_CHECK(
      keep_value <= segment_items_value,
      "n16+n8+n4 must not exceed R*R");
  TORCH_CHECK(
      segment_items_value <= std::numeric_limits<int>::max(),
      "R*R exceeds CUB int32 range");
  const int64_t segments_value = checked_positive_product(
      probability_fp16.size(0),
      probability_fp16.size(1),
      "B*H sort segments");
  TORCH_CHECK(
      segments_value <= std::numeric_limits<int>::max(),
      "B*H exceeds CUB int32 range");
  if (fp16_blocks_by_head != nullptr) {
    TORCH_CHECK(
        fp16_blocks_by_head->defined() && fp16_blocks_by_head->is_cuda(),
        "fp16_blocks_by_head must be a CUDA tensor");
    TORCH_CHECK(
        fp16_blocks_by_head->device() == probability_fp16.device(),
        "fp16_blocks_by_head must share the score device");
    TORCH_CHECK(
        fp16_blocks_by_head->scalar_type() == at::ScalarType::Int &&
            fp16_blocks_by_head->is_contiguous(),
        "fp16_blocks_by_head must be contiguous int32");
    TORCH_CHECK(
        fp16_blocks_by_head->dim() == 2 &&
            fp16_blocks_by_head->size(0) == probability_fp16.size(0) &&
            fp16_blocks_by_head->size(1) == probability_fp16.size(1),
        "fp16_blocks_by_head must have shape [B,H]");
    TORCH_CHECK(
        n4 == 0,
        "per-head FP16 redistribution supports FP8/FP16 retained phases only");
  }
  const int64_t total_items_value = checked_positive_product(
      segments_value, segment_items_value, "total sort items");
  TORCH_CHECK(
      total_items_value <= std::numeric_limits<int>::max(),
      "B*H*R*R exceeds CUB int32 range");
  const int64_t row_count_value = checked_positive_product(
      segments_value, rows, "B*H*query rows");
  TORCH_CHECK(
      row_count_value <= std::numeric_limits<int>::max(),
      "B*H*R exceeds int32 row-sentinel range");

  c10::cuda::CUDAGuard device_guard(probability_fp16.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(probability_fp16.get_device());
  TORCH_CHECK(
      (properties->major == 8 && properties->minor == 9) ||
          (properties->major == 12 && properties->minor == 0),
      "mpa._cuda_router requires SM89 or SM120, found sm_",
      properties->major, properties->minor);

  const int segment_items = static_cast<int>(segment_items_value);
  const int segments = static_cast<int>(segments_value);
  const int total_items = static_cast<int>(total_items_value);
  const int row_count = static_cast<int>(row_count_value);
  const int keep = static_cast<int>(keep_value);
  const int n16_end = n16;
  const int n8_end = n16 + n8;
  const bool use_global_sort =
      request_global_sort && segments <= kMaxCompositeSegments;
  TORCH_CHECK(
      spatial_cross_blocks_per_frame == 0 || use_global_sort,
      "spatial-cross routing requires the exact global radix-sort path");
  auto int_options = probability_fp16.options().dtype(at::ScalarType::Int);
  auto byte_options = probability_fp16.options().dtype(at::ScalarType::Byte);
  auto precision_map = torch::empty(probability_fp16.sizes(), byte_options);
  auto packed_ids = torch::empty(probability_fp16.sizes(), int_options);
  auto fp8_counts = torch::empty(
      {probability_fp16.size(0), probability_fp16.size(1), rows},
      int_options);
  auto fp16_counts = torch::empty_like(fp8_counts);
  auto fp4_counts = torch::empty_like(fp8_counts);
  auto first_empty_row = torch::full({1}, -1, int_options);

  auto input_composite_keys = use_global_sort
      ? torch::empty(probability_fp16.sizes(), int_options)
      : torch::empty({0}, int_options);
  auto sorted_composite_keys = use_global_sort
      ? torch::empty(probability_fp16.sizes(), int_options)
      : torch::empty({0}, int_options);
  auto sorted_half_keys = use_global_sort
      ? torch::empty({0}, probability_fp16.options())
      : torch::empty_like(probability_fp16);
  auto sorted_ids = torch::empty_like(packed_ids);
  auto segment_offsets = use_global_sort
      ? torch::empty({0}, int_options)
      : torch::empty({segments_value + 1}, int_options);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(probability_fp16.get_device());
  const int64_t initialization_items = use_global_sort
      ? total_items_value
      : std::max(total_items_value, segments_value + 1);
  const int64_t initialization_blocks =
      (initialization_items + kThreads - 1) / kThreads;
  TORCH_CHECK(
      initialization_blocks <= kMaxGridX,
      "routing initialization grid.x exceeds CUDA limit");
  if (use_global_sort) {
    initialize_composite_sort_kernel<<<
        static_cast<unsigned int>(initialization_blocks),
        kThreads,
        0,
        stream>>>(
        reinterpret_cast<const half*>(
            probability_fp16.data_ptr<at::Half>()),
        reinterpret_cast<uint32_t*>(input_composite_keys.data_ptr<int>()),
        packed_ids.data_ptr<int>(),
        total_items,
        segment_items,
        static_cast<int>(columns),
        spatial_cross_blocks_per_frame,
        spatial_cross_patches_w);
  } else {
    initialize_sort_metadata_kernel<<<
        static_cast<unsigned int>(initialization_blocks),
        kThreads,
        0,
        stream>>>(
        packed_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        total_items,
        segments,
        segment_items);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t sort_workspace_bytes = 0;
  const int composite_key_bits =
      kFullFp16KeyBits + required_unsigned_bits(segments);
  if (use_global_sort) {
    cub::DoubleBuffer<uint32_t> query_keys(
        reinterpret_cast<uint32_t*>(input_composite_keys.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(sorted_composite_keys.data_ptr<int>()));
    cub::DoubleBuffer<int> query_values(
        packed_ids.data_ptr<int>(), sorted_ids.data_ptr<int>());
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_workspace_bytes,
        query_keys,
        query_values,
        total_items,
        0,
        composite_key_bits,
        stream));
  } else {
    C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        sort_workspace_bytes,
        reinterpret_cast<const half*>(
            probability_fp16.data_ptr<at::Half>()),
        reinterpret_cast<half*>(sorted_half_keys.data_ptr<at::Half>()),
        packed_ids.data_ptr<int>(),
        sorted_ids.data_ptr<int>(),
        total_items,
        segments,
        segment_offsets.data_ptr<int>(),
        segment_offsets.data_ptr<int>() + 1,
        0,
        kFullFp16KeyBits,
        stream));
  }
  TORCH_CHECK(
      sort_workspace_bytes <=
          static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "CUB routing workspace exceeds int64 tensor range");
  auto sort_workspace = torch::empty(
      {static_cast<int64_t>(sort_workspace_bytes)}, byte_options);
  const int* selected_sorted_ids = sorted_ids.data_ptr<int>();
  if (use_global_sort) {
    cub::DoubleBuffer<uint32_t> sort_keys(
        reinterpret_cast<uint32_t*>(input_composite_keys.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(sorted_composite_keys.data_ptr<int>()));
    cub::DoubleBuffer<int> sort_values(
        packed_ids.data_ptr<int>(), sorted_ids.data_ptr<int>());
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_workspace.data_ptr<uint8_t>(),
        sort_workspace_bytes,
        sort_keys,
        sort_values,
        total_items,
        0,
        composite_key_bits,
        stream));
    selected_sorted_ids = sort_values.Current();
  } else {
    C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        sort_workspace.data_ptr<uint8_t>(),
        sort_workspace_bytes,
        reinterpret_cast<const half*>(
            probability_fp16.data_ptr<at::Half>()),
        reinterpret_cast<half*>(sorted_half_keys.data_ptr<at::Half>()),
        packed_ids.data_ptr<int>(),
        sorted_ids.data_ptr<int>(),
        total_items,
        segments,
        segment_offsets.data_ptr<int>(),
        segment_offsets.data_ptr<int>() + 1,
        0,
        kFullFp16KeyBits,
        stream));
  }

  const int64_t item_blocks =
      (total_items_value + kThreads - 1) / kThreads;
  TORCH_CHECK(
      item_blocks <= kMaxGridX,
      "routing scatter grid.x exceeds CUDA limit");
  scatter_precision_kernel<<<
      static_cast<unsigned int>(item_blocks),
      kThreads,
      0,
      stream>>>(
      selected_sorted_ids,
      precision_map.data_ptr<uint8_t>(),
      fp16_blocks_by_head == nullptr
          ? nullptr
          : fp16_blocks_by_head->data_ptr<int>(),
      total_items,
      segment_items,
      n16_end,
      n8_end,
      keep);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  TORCH_CHECK(
      row_count_value <= kMaxGridX,
      "routing row-pack grid.x exceeds CUDA limit");
  pack_precision_rows_kernel<<<
      static_cast<unsigned int>(row_count_value),
      kThreads,
      0,
      stream>>>(
      precision_map.data_ptr<uint8_t>(),
      packed_ids.data_ptr<int>(),
      fp8_counts.data_ptr<int>(),
      fp16_counts.data_ptr<int>(),
      fp4_counts.data_ptr<int>(),
      first_empty_row.data_ptr<int>(),
      row_count,
      static_cast<int>(columns));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      precision_map,
      packed_ids,
      fp8_counts,
      fp16_counts,
      fp4_counts,
      first_empty_row};
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision(
    torch::Tensor probability_fp16,
    int64_t n16_value,
    int64_t n8_value,
    int64_t n4_value) {
  TORCH_CHECK(
      probability_fp16.dim() == 4 &&
          probability_fp16.size(2) == probability_fp16.size(3),
      "probability_fp16 must be square in its final two dimensions");
  return route_precision_impl(
      std::move(probability_fp16),
      n16_value,
      n8_value,
      n4_value,
      true);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision_spatial_cross(
    torch::Tensor probability_fp16,
    int64_t n16_value,
    int64_t n8_value,
    int64_t n4_value,
    int64_t frames_value,
    int64_t patches_h_value,
    int64_t patches_w_value) {
  check_probability(probability_fp16);
  TORCH_CHECK(
      probability_fp16.size(2) == probability_fp16.size(3),
      "probability_fp16 must be square in its final two dimensions");
  const int frames = require_nonnegative_int32(frames_value, "frames");
  const int patches_h = require_nonnegative_int32(patches_h_value, "patches_h");
  const int patches_w = require_nonnegative_int32(patches_w_value, "patches_w");
  TORCH_CHECK(
      frames > 0 && patches_h > 0 && patches_w > 0,
      "spatial-cross geometry must be positive");
  const int64_t blocks_per_frame_value = checked_positive_product(
      patches_h_value, patches_w_value, "2-D blocks per frame");
  const int64_t blocks_value = checked_positive_product(
      frames_value, blocks_per_frame_value, "2-D route blocks");
  TORCH_CHECK(
      probability_fp16.size(2) == blocks_value,
      "spatial-cross geometry must match the probability dimensions");
  // Both products are bounded by blocks_per_frame_value and may legitimately
  // be zero for a one-row or one-column logical grid.
  const int64_t horizontal_edges =
      patches_h_value * (patches_w_value - 1);
  const int64_t vertical_edges =
      (patches_h_value - 1) * patches_w_value;
  const int64_t anchors_per_frame = blocks_per_frame_value +
      2 * (horizontal_edges + vertical_edges);
  const int64_t anchor_count = frames_value * anchors_per_frame;
  TORCH_CHECK(
      n16_value + n8_value + n4_value >= anchor_count,
      "mandatory spatial-cross anchors exceed the retained global budget");
  return route_precision_impl(
      std::move(probability_fp16),
      n16_value,
      n8_value,
      n4_value,
      true,
      nullptr,
      static_cast<int>(blocks_per_frame_value),
      patches_w);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_sol_scores(
    torch::Tensor scores_fp16,
    double beta_value,
    double retained_fp16_ratio,
    double retained_fp8_ratio,
    double retained_fp4_ratio,
    bool force_dense) {
  check_probability(scores_fp16);
  TORCH_CHECK(std::isfinite(beta_value), "beta must be finite");
  TORCH_CHECK(
      std::isfinite(retained_fp16_ratio) && retained_fp16_ratio >= 0.0 &&
          std::isfinite(retained_fp8_ratio) && retained_fp8_ratio >= 0.0 &&
          std::isfinite(retained_fp4_ratio) && retained_fp4_ratio >= 0.0,
      "retained precision ratios must be finite and nonnegative");
  const double ratio_total =
      retained_fp16_ratio + retained_fp8_ratio + retained_fp4_ratio;
  TORCH_CHECK(
      std::abs(ratio_total - 1.0) <= 1.0e-6,
      "retained precision ratios must sum to one");
  const double ratio16 = retained_fp16_ratio / ratio_total;
  const double ratio8 = retained_fp8_ratio / ratio_total;
  const double ratio4 = retained_fp4_ratio / ratio_total;

  const int64_t rows_value = scores_fp16.size(2);
  const int64_t segment_items_value = checked_positive_product(
      rows_value, rows_value, "Sol sort items per head");
  const int64_t segments_value = checked_positive_product(
      scores_fp16.size(0), scores_fp16.size(1), "Sol B*H segments");
  const int64_t total_items_value = checked_positive_product(
      segments_value, segment_items_value, "Sol total sort items");
  const int64_t row_count_value = checked_positive_product(
      segments_value, rows_value, "Sol B*H*query rows");
  TORCH_CHECK(
      rows_value <= std::numeric_limits<int>::max() &&
          segment_items_value <= std::numeric_limits<int>::max() &&
          segments_value <= std::numeric_limits<int>::max() &&
          total_items_value <= std::numeric_limits<int>::max() &&
          row_count_value <= std::numeric_limits<int>::max(),
      "Sol routing geometry exceeds its int32 domain");

  c10::cuda::CUDAGuard device_guard(scores_fp16.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(scores_fp16.get_device());
  TORCH_CHECK(
      (properties->major == 8 && properties->minor == 9) ||
          (properties->major == 12 && properties->minor == 0),
      "Sol routing requires SM89 or SM120, found sm_",
      properties->major, properties->minor);

  const int rows = static_cast<int>(rows_value);
  const int segment_items = static_cast<int>(segment_items_value);
  const int segments = static_cast<int>(segments_value);
  const int total_items = static_cast<int>(total_items_value);
  const int row_count = static_cast<int>(row_count_value);
  auto int_options = scores_fp16.options().dtype(at::ScalarType::Int);
  auto byte_options = scores_fp16.options().dtype(at::ScalarType::Byte);
  auto float_options = scores_fp16.options().dtype(at::ScalarType::Float);
  auto normalized_scores = torch::empty(scores_fp16.sizes(), float_options);
  auto sorted_scores = torch::empty_like(normalized_scores);
  auto precision_map = torch::empty(scores_fp16.sizes(), byte_options);
  auto packed_ids = torch::empty(scores_fp16.sizes(), int_options);
  auto sorted_ids = torch::empty_like(packed_ids);
  auto segment_offsets = torch::empty({segments_value + 1}, int_options);
  auto selected_per_segment = torch::zeros({segments_value}, int_options);
  auto phase_totals = torch::empty({segments_value, 3}, int_options);
  auto fp8_counts = torch::empty(
      {scores_fp16.size(0), scores_fp16.size(1), rows_value}, int_options);
  auto fp16_counts = torch::empty_like(fp8_counts);
  auto fp4_counts = torch::empty_like(fp8_counts);
  auto first_empty_row = torch::full({1}, -1, int_options);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(scores_fp16.get_device());

  TORCH_CHECK(row_count_value <= kMaxGridX, "Sol row grid exceeds CUDA limit");
  normalize_sol_rows_kernel<<<
      static_cast<unsigned int>(row_count_value), kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(scores_fp16.data_ptr<at::Half>()),
      normalized_scores.data_ptr<float>(), selected_per_segment.data_ptr<int>(),
      row_count, rows, static_cast<float>(beta_value), force_dense);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t initialization_items =
      std::max(total_items_value, segments_value + 1);
  const int64_t initialization_blocks =
      (initialization_items + kThreads - 1) / kThreads;
  TORCH_CHECK(
      initialization_blocks <= kMaxGridX,
      "Sol sort initialization grid exceeds CUDA limit");
  initialize_sort_metadata_kernel<<<
      static_cast<unsigned int>(initialization_blocks), kThreads, 0, stream>>>(
      packed_ids.data_ptr<int>(), segment_offsets.data_ptr<int>(), total_items,
      segments, segment_items);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int phase_blocks = (segments + kThreads - 1) / kThreads;
  allocate_sol_phase_totals_kernel<<<phase_blocks, kThreads, 0, stream>>>(
      selected_per_segment.data_ptr<int>(), phase_totals.data_ptr<int>(),
      segments, ratio16, ratio8, ratio4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t sort_workspace_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, sort_workspace_bytes, normalized_scores.data_ptr<float>(),
      sorted_scores.data_ptr<float>(), packed_ids.data_ptr<int>(),
      sorted_ids.data_ptr<int>(), total_items, segments,
      segment_offsets.data_ptr<int>(), segment_offsets.data_ptr<int>() + 1,
      0, 32, stream));
  TORCH_CHECK(
      sort_workspace_bytes <=
          static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "Sol sort workspace exceeds int64 tensor range");
  auto sort_workspace = torch::empty(
      {static_cast<int64_t>(sort_workspace_bytes)}, byte_options);
  C10_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      sort_workspace.data_ptr<uint8_t>(), sort_workspace_bytes,
      normalized_scores.data_ptr<float>(), sorted_scores.data_ptr<float>(),
      packed_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(), total_items,
      segments, segment_offsets.data_ptr<int>(),
      segment_offsets.data_ptr<int>() + 1, 0, 32, stream));

  const int item_blocks = (total_items + kThreads - 1) / kThreads;
  scatter_sol_precision_kernel<<<item_blocks, kThreads, 0, stream>>>(
      sorted_ids.data_ptr<int>(), selected_per_segment.data_ptr<int>(),
      phase_totals.data_ptr<int>(), precision_map.data_ptr<uint8_t>(),
      total_items, segment_items);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  pack_precision_rows_kernel<<<
      static_cast<unsigned int>(row_count_value), kThreads, 0, stream>>>(
      precision_map.data_ptr<uint8_t>(), packed_ids.data_ptr<int>(),
      fp8_counts.data_ptr<int>(), fp16_counts.data_ptr<int>(),
      fp4_counts.data_ptr<int>(), first_empty_row.data_ptr<int>(), row_count,
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      precision_map, packed_ids, fp8_counts, fp16_counts, fp4_counts,
      first_empty_row};
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision_head_fp16(
    torch::Tensor probability_fp16,
    torch::Tensor fp16_blocks_by_head,
    int64_t keep_value) {
  TORCH_CHECK(
      probability_fp16.dim() == 4 &&
          probability_fp16.size(2) == probability_fp16.size(3),
      "probability_fp16 must be square in its final two dimensions");
  const int keep = require_nonnegative_int32(keep_value, "keep");
  TORCH_CHECK(keep > 0, "keep must be positive");
  return route_precision_impl(
      std::move(probability_fp16),
      0,
      keep,
      0,
      true,
      &fp16_blocks_by_head);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_sol_threshold(
    torch::Tensor logits_fp16,
    torch::Tensor q_pool_fp16,
    torch::Tensor k_pool_fp16,
    torch::Tensor prefix_k_fp16,
    torch::Tensor prefix_counts_int32,
    double low8_tau_value,
    double fp16_tau_value,
    int64_t threshold_type_value,
    int64_t local_fp16_radius_value,
    int64_t forced_sink_blocks_value,
    int64_t frames_value,
    int64_t height_value,
    int64_t width_value,
    int64_t raster_pool_width_value,
    bool partial_logmass) {
  check_probability(logits_fp16);
  check_sol_pool_tensor(q_pool_fp16, "q_pool_fp16");
  check_sol_pool_tensor(k_pool_fp16, "k_pool_fp16");
  TORCH_CHECK(
      logits_fp16.size(2) == logits_fp16.size(3),
      "logits_fp16 must be square in its final two dimensions");
  const int64_t batch = logits_fp16.size(0);
  const int64_t q_heads = logits_fp16.size(1);
  const int64_t rows = logits_fp16.size(2);
  const int64_t kv_heads = k_pool_fp16.size(1);
  const int64_t head_dim = q_pool_fp16.size(3);
  const bool has_prefix_statistics = prefix_k_fp16.numel() != 0;
  TORCH_CHECK(
      q_pool_fp16.device() == logits_fp16.device() &&
          k_pool_fp16.device() == logits_fp16.device(),
      "logits_fp16, q_pool_fp16, and k_pool_fp16 must share one CUDA device");
  TORCH_CHECK(
      q_pool_fp16.size(0) == batch && q_pool_fp16.size(1) == q_heads &&
          q_pool_fp16.size(2) == rows,
      "q_pool_fp16 must have shape [B,Hq,R,D] matching logits_fp16");
  TORCH_CHECK(
      k_pool_fp16.size(0) == batch && k_pool_fp16.size(2) == rows &&
          k_pool_fp16.size(3) == head_dim,
      "k_pool_fp16 must have shape [B,Hkv,R,D] matching q_pool_fp16");
  TORCH_CHECK(
      q_heads % kv_heads == 0,
      "Q pool heads must be divisible by KV pool heads");
  TORCH_CHECK(
      prefix_k_fp16.defined() && prefix_counts_int32.defined(),
      "prefix statistic tensors must be defined");
  TORCH_CHECK(
      prefix_k_fp16.is_cuda() && prefix_counts_int32.is_cuda() &&
          prefix_k_fp16.device() == logits_fp16.device() &&
          prefix_counts_int32.device() == logits_fp16.device(),
      "prefix statistic tensors must share the route CUDA device");
  TORCH_CHECK(
      prefix_k_fp16.is_contiguous() && prefix_counts_int32.is_contiguous(),
      "prefix statistic tensors must be contiguous");
  TORCH_CHECK(
      prefix_k_fp16.scalar_type() == at::ScalarType::Half,
      "prefix_k_fp16 must have dtype torch.float16");
  TORCH_CHECK(
      prefix_counts_int32.scalar_type() == at::ScalarType::Int,
      "prefix_counts_int32 must have dtype torch.int32");
  if (has_prefix_statistics) {
    TORCH_CHECK(
        prefix_k_fp16.dim() == 4 && prefix_k_fp16.size(0) == batch &&
            prefix_k_fp16.size(1) == kv_heads &&
            prefix_k_fp16.size(2) > 0 && prefix_k_fp16.size(3) == head_dim,
        "prefix_k_fp16 must have shape [B,Hkv,T_capacity,D]");
    TORCH_CHECK(
        prefix_counts_int32.dim() == 1 &&
            prefix_counts_int32.size(0) == batch,
        "prefix_counts_int32 must have shape [B]");
    TORCH_CHECK(
        threshold_type_value == 0,
        "prefix-inclusive K statistics currently require threshold_type=diag");
  } else {
    TORCH_CHECK(
        prefix_k_fp16.dim() == 1 && prefix_k_fp16.size(0) == 0 &&
            prefix_counts_int32.dim() == 1 &&
            prefix_counts_int32.size(0) == 0,
        "disabled prefix statistics require empty rank-one sentinels");
  }
  TORCH_CHECK(
      std::isfinite(low8_tau_value) && std::isfinite(fp16_tau_value),
      "Sol route tau values must be finite");
  TORCH_CHECK(
      fp16_tau_value >= low8_tau_value,
      "fp16_tau must be greater than or equal to low8_tau");
  TORCH_CHECK(
      threshold_type_value == 0 || threshold_type_value == 1,
      "threshold_type must be 0 (diag) or 1 (exact)");
  const int local_fp16_radius = require_nonnegative_int32(
      local_fp16_radius_value, "local_fp16_radius");
  const int forced_sink_blocks = require_nonnegative_int32(
      forced_sink_blocks_value, "forced_sink_blocks");
  const int frames = require_nonnegative_int32(frames_value, "frames");
  const int height = require_nonnegative_int32(height_value, "height");
  const int width = require_nonnegative_int32(width_value, "width");
  const int raster_pool_width = require_nonnegative_int32(
      raster_pool_width_value, "raster_pool_width");
  TORCH_CHECK(frames > 0 && height > 0 && width > 0,
              "Sol route raster geometry must be positive");
  TORCH_CHECK(
      forced_sink_blocks <= rows,
      "forced_sink_blocks cannot exceed the routed key-block count");
  TORCH_CHECK(
      raster_pool_width == 8 || raster_pool_width == 16,
      "raster_pool_width must be 8 or 16");
  const int64_t expected_video_rows = checked_positive_product(
      checked_positive_product(
          frames, (height + 7) / 8, "Sol route F*ceil(H/8)"),
      (width + raster_pool_width - 1) / raster_pool_width,
      "Sol route raster patch count");
  TORCH_CHECK(
      expected_video_rows == rows,
      "Sol route pooled rows do not match raster geometry");
  TORCH_CHECK(
      !partial_logmass || !has_prefix_statistics,
      "partial log-mass routing is currently incompatible with prefix K "
      "statistics");
  TORCH_CHECK(
      batch <= std::numeric_limits<int>::max() &&
          q_heads <= std::numeric_limits<int>::max() &&
          kv_heads <= std::numeric_limits<int>::max() &&
          rows <= std::numeric_limits<int>::max() &&
          (!has_prefix_statistics ||
           prefix_k_fp16.size(2) <= std::numeric_limits<int>::max()) &&
          head_dim <= std::numeric_limits<int>::max(),
      "Sol route dimensions exceed int32 range");
  const int64_t row_count_value = checked_positive_product(
      checked_positive_product(batch, q_heads, "Sol route B*Hq"),
      rows,
      "Sol route B*Hq*R");
  const int64_t total_items_value = checked_positive_product(
      row_count_value, rows, "Sol route B*Hq*R*R");
  TORCH_CHECK(
      row_count_value <= kMaxGridX &&
          row_count_value <= std::numeric_limits<int>::max() &&
          total_items_value <= std::numeric_limits<int>::max(),
      "Sol route item domain exceeds CUDA/int32 limits");

  c10::cuda::CUDAGuard device_guard(logits_fp16.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(logits_fp16.get_device());
  TORCH_CHECK(
      (properties->major == 8 && properties->minor == 9) ||
          (properties->major == 12 && properties->minor == 0),
      "mpa._cuda_router requires SM89 or SM120, found sm_",
      properties->major, properties->minor);

  auto int_options = logits_fp16.options().dtype(at::ScalarType::Int);
  auto byte_options = logits_fp16.options().dtype(at::ScalarType::Byte);
  auto float_options = logits_fp16.options().dtype(at::ScalarType::Float);
  auto precision_map = torch::empty(logits_fp16.sizes(), byte_options);
  auto packed_ids = torch::empty(logits_fp16.sizes(), int_options);
  auto fp8_counts = torch::empty({batch, q_heads, rows}, int_options);
  auto fp16_counts = torch::empty_like(fp8_counts);
  auto fp4_counts = torch::empty_like(fp8_counts);
  auto first_empty_row = torch::full({1}, -1, int_options);
  auto k_mean = threshold_type_value == 0
      ? torch::empty({batch, kv_heads, head_dim}, float_options)
      : torch::empty({0}, float_options);
  auto k_variance = threshold_type_value == 0
      ? torch::empty_like(k_mean)
      : torch::empty({0}, float_options);
  auto k_logmass_covariance = threshold_type_value == 0 && partial_logmass
      ? torch::empty_like(k_mean)
      : torch::empty({0}, float_options);
  auto logmass_mean = threshold_type_value == 0 && partial_logmass
      ? torch::empty({batch, kv_heads}, float_options)
      : torch::empty({0}, float_options);
  auto logmass_variance = threshold_type_value == 0 && partial_logmass
      ? torch::empty_like(logmass_mean)
      : torch::empty({0}, float_options);

  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(logits_fp16.get_device());
  if (threshold_type_value == 0) {
    const int64_t owners_value = checked_positive_product(
        batch, kv_heads, "Sol route B*Hkv");
    TORCH_CHECK(owners_value <= kMaxGridX, "Sol K-stat grid.x exceeds CUDA limit");
    sol_k_diag_stats_kernel<<<
        static_cast<unsigned int>(owners_value), kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(k_pool_fp16.data_ptr<at::Half>()),
        has_prefix_statistics
            ? reinterpret_cast<const half*>(
                  prefix_k_fp16.data_ptr<at::Half>())
            : nullptr,
        has_prefix_statistics
            ? prefix_counts_int32.data_ptr<int32_t>()
            : nullptr,
        k_mean.data_ptr<float>(),
        k_variance.data_ptr<float>(),
        partial_logmass ? k_logmass_covariance.data_ptr<float>() : nullptr,
        partial_logmass ? logmass_mean.data_ptr<float>() : nullptr,
        partial_logmass ? logmass_variance.data_ptr<float>() : nullptr,
        static_cast<int>(owners_value),
        static_cast<int>(kv_heads),
        static_cast<int>(rows),
        has_prefix_statistics
            ? static_cast<int>(prefix_k_fp16.size(2))
            : 0,
        static_cast<int>(head_dim),
        height,
        width,
        raster_pool_width,
        partial_logmass);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    sol_threshold_route_kernel<false><<<
        static_cast<unsigned int>(row_count_value), kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(logits_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_pool_fp16.data_ptr<at::Half>()),
        k_mean.data_ptr<float>(),
        k_variance.data_ptr<float>(),
        partial_logmass ? k_logmass_covariance.data_ptr<float>() : nullptr,
        partial_logmass ? logmass_mean.data_ptr<float>() : nullptr,
        partial_logmass ? logmass_variance.data_ptr<float>() : nullptr,
        precision_map.data_ptr<uint8_t>(),
        static_cast<int>(row_count_value),
        static_cast<int>(q_heads),
        static_cast<int>(kv_heads),
        static_cast<int>(rows),
        static_cast<int>(head_dim),
        static_cast<float>(low8_tau_value),
        static_cast<float>(fp16_tau_value),
        local_fp16_radius,
        forced_sink_blocks,
        height,
        width,
        raster_pool_width,
        partial_logmass);
  } else {
    sol_threshold_route_kernel<true><<<
        static_cast<unsigned int>(row_count_value), kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(logits_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_pool_fp16.data_ptr<at::Half>()),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        precision_map.data_ptr<uint8_t>(),
        static_cast<int>(row_count_value),
        static_cast<int>(q_heads),
        static_cast<int>(kv_heads),
        static_cast<int>(rows),
        static_cast<int>(head_dim),
        static_cast<float>(low8_tau_value),
        static_cast<float>(fp16_tau_value),
        local_fp16_radius,
        forced_sink_blocks,
        height,
        width,
        raster_pool_width,
        partial_logmass);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  pack_precision_rows_kernel<<<
      static_cast<unsigned int>(row_count_value), kThreads, 0, stream>>>(
      precision_map.data_ptr<uint8_t>(),
      packed_ids.data_ptr<int>(),
      fp8_counts.data_ptr<int>(),
      fp16_counts.data_ptr<int>(),
      fp4_counts.data_ptr<int>(),
      first_empty_row.data_ptr<int>(),
      static_cast<int>(row_count_value),
      static_cast<int>(rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      precision_map,
      packed_ids,
      fp8_counts,
      fp16_counts,
      fp4_counts,
      first_empty_row};
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision_k64(
    torch::Tensor score_fp16,
    int64_t n16_value,
    int64_t n8_value,
    int64_t n4_value) {
  TORCH_CHECK(
      score_fp16.dim() == 4 &&
          score_fp16.size(3) == 2 * score_fp16.size(2),
      "K64 route score must have shape [B,H,R,2R]");
  return route_precision_impl(
      std::move(score_fp16), n16_value, n8_value, n4_value, true);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision_segmented_control(
    torch::Tensor probability_fp16,
    int64_t n16_value,
    int64_t n8_value,
    int64_t n4_value) {
  return route_precision_impl(
      std::move(probability_fp16),
      n16_value,
      n8_value,
      n4_value,
      false);
}

torch::Tensor validate_route_metadata(
    torch::Tensor fp8_counts,
    torch::Tensor fp16_counts,
    torch::Tensor fp4_counts,
    torch::Tensor first_empty_row,
    int64_t n8_value,
    int64_t n16_value,
    int64_t n4_value) {
  check_count_tensor(fp8_counts, fp8_counts, "fp8_counts");
  check_count_tensor(fp16_counts, fp8_counts, "fp16_counts");
  check_count_tensor(fp4_counts, fp8_counts, "fp4_counts");
  TORCH_CHECK(
      fp8_counts.size(0) > 0 && fp8_counts.size(1) > 0 &&
          fp8_counts.size(2) > 0,
      "route count dimensions must be positive");
  TORCH_CHECK(
      first_empty_row.defined() && first_empty_row.is_cuda() &&
          first_empty_row.is_contiguous() &&
          first_empty_row.scalar_type() == at::ScalarType::Int &&
          first_empty_row.numel() == 1,
      "first_empty_row must be one contiguous CUDA int32 value");
  TORCH_CHECK(
      first_empty_row.device() == fp8_counts.device(),
      "first_empty_row device must match route counts");
  const int n8 = require_nonnegative_int32(n8_value, "n8");
  const int n16 = require_nonnegative_int32(n16_value, "n16");
  const int n4 = require_nonnegative_int32(n4_value, "n4");
  const int64_t rows_value = fp8_counts.size(2);
  const int64_t segment_items_value =
      checked_positive_product(rows_value, rows_value, "query-pair count");
  TORCH_CHECK(
      n8_value + n16_value + n4_value <= segment_items_value,
      "route phase counts must not exceed R*R");
  const int64_t segments_value = checked_positive_product(
      fp8_counts.size(0), fp8_counts.size(1), "B*H validation segments");
  const int64_t row_count_value = checked_positive_product(
      segments_value, rows_value, "B*H*query validation rows");
  TORCH_CHECK(
      row_count_value <= std::numeric_limits<int>::max(),
      "row count exceeds int32 sentinel range");

  c10::cuda::CUDAGuard device_guard(fp8_counts.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(fp8_counts.get_device());
  TORCH_CHECK(
      (properties->major == 8 && properties->minor == 9) ||
          (properties->major == 12 && properties->minor == 0),
      "mpa._cuda_router requires SM89 or SM120, found sm_",
      properties->major, properties->minor);
  auto validation = torch::empty({3}, fp8_counts.options());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(fp8_counts.get_device());
  validate_route_metadata_kernel<<<1, kThreads, 0, stream>>>(
      fp8_counts.data_ptr<int>(),
      fp16_counts.data_ptr<int>(),
      fp4_counts.data_ptr<int>(),
      first_empty_row.data_ptr<int>(),
      validation.data_ptr<int>(),
      static_cast<int>(segments_value),
      static_cast<int>(rows_value),
      n8,
      n16,
      n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return validation;
}
