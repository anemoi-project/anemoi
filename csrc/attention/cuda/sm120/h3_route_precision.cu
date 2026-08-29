/* Native global route for the SM120 MiniMax-H3 production path. */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <tuple>

#include "api.h"

// The stable CUB route and compact-anchor algorithm is shared source, while
// each extension keeps an architecture-owned entry point and launch tuning.
// SM120 remains the default instantiation; the SM89 translation unit supplies
// its own names, capability predicate, and implicit-FP16-prefix convention.
#ifndef MPA_ROUTE_PRECISION_FUNCTION
#define MPA_ROUTE_PRECISION_FUNCTION sm120_h3_route_precision
#endif
#ifndef MPA_MATERIALIZE_ROUTE_FUNCTION
#define MPA_MATERIALIZE_ROUTE_FUNCTION sm120_h3_materialize_route
#endif
#ifndef MPA_ROUTE_DEVICE_OK
#define MPA_ROUTE_DEVICE_OK(properties) \
  ((properties)->major == 12 && (properties)->minor == 0)
#endif
#ifndef MPA_ROUTE_DEVICE_LABEL
#define MPA_ROUTE_DEVICE_LABEL "SM120"
#endif
#ifndef MPA_ROUTE_THREADS
#define MPA_ROUTE_THREADS 256
#endif
#ifndef MPA_IMPLICIT_HIGH_PREFIX
#define MPA_IMPLICIT_HIGH_PREFIX 0
#endif

namespace {

constexpr int kThreads = MPA_ROUTE_THREADS;
constexpr int kProbabilityBits = 16;
constexpr int kMaxSegments = 1 << 16;
constexpr int64_t kMaxGridX = 2147483647LL;
constexpr uint8_t kSkip = 0;
constexpr uint8_t kLow = 1;
constexpr uint8_t kMiddle = 2;
constexpr uint8_t kHigh = 3;

struct PhaseCounts {
  int middle;
  int high;
  int low;
};

struct AddPhaseCounts {
  __device__ __forceinline__ PhaseCounts operator()(
      const PhaseCounts& lhs, const PhaseCounts& rhs) const {
    return {
        lhs.middle + rhs.middle,
        lhs.high + rhs.high,
        lhs.low + rhs.low};
  }
};

using RowReduce = cub::BlockReduce<PhaseCounts, kThreads>;
using RowScan = cub::BlockScan<PhaseCounts, kThreads>;
using AnchorReduce = cub::BlockReduce<int, kThreads>;
using AnchorScan = cub::BlockScan<int, kThreads>;

union RowTempStorage {
  typename RowReduce::TempStorage reduce;
  typename RowScan::TempStorage scan;
};

union AnchorTempStorage {
  typename AnchorReduce::TempStorage reduce;
  typename AnchorScan::TempStorage scan;
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
  int bits = 0;
  for (int value = value_count - 1; value > 0; value >>= 1) {
    ++bits;
  }
  return bits;
}

__global__ void initialize_composite_sort_kernel(
    const half* __restrict__ probability,
    uint32_t* __restrict__ composite_keys,
    int* __restrict__ initial_ids,
    int total_items,
    int segment_items) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_items) {
    return;
  }
  const int segment = static_cast<int>(index) / segment_items;
  const int flat_id = static_cast<int>(index) - segment * segment_items;
  const uint16_t probability_bits =
      __half_as_ushort(probability[index]) & uint16_t{0x7fff};
  const uint32_t descending_probability =
      uint32_t{0xffff} - static_cast<uint32_t>(probability_bits);
  composite_keys[index] =
      (static_cast<uint32_t>(segment) << kProbabilityBits) |
      descending_probability;
  initial_ids[index] = flat_id;
}

__global__ void scatter_precision_kernel(
    const int* __restrict__ sorted_ids,
    uint8_t* __restrict__ precision_map,
    int total_items,
    int segment_items,
    int high_end,
    int middle_end,
    int keep) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_items) {
    return;
  }
  const int rank = static_cast<int>(linear) % segment_items;
  const int segment = static_cast<int>(linear) / segment_items;
  uint8_t code = kSkip;
  if (rank < high_end) {
    code = kHigh;
  } else if (rank < middle_end) {
    code = kMiddle;
  } else if (rank < keep) {
    code = kLow;
  }
  precision_map[segment * segment_items + sorted_ids[linear]] = code;
}

__global__ __launch_bounds__(kThreads) void apply_anchor_budget_kernel(
    const int* __restrict__ sorted_ids,
    const bool* __restrict__ anchors,
    const int* __restrict__ anchor_ids,
    uint8_t* __restrict__ precision_map,
    int anchor_count,
    int segment_items,
    int low_begin,
    int keep,
    uint8_t low_code) {
  const int segment = static_cast<int>(blockIdx.x);
  const int segment_offset = segment * segment_items;
  __shared__ AnchorTempStorage temp;
  __shared__ int missing_total;
  __shared__ int evicted;

  int local_missing = 0;
  for (int id_index = threadIdx.x; id_index < anchor_count;
       id_index += blockDim.x) {
    const int id = anchor_ids[id_index];
    const int index = segment_offset + id;
    const bool missing = precision_map[index] == kSkip;
    if (missing) {
      precision_map[index] = low_code;
      ++local_missing;
    }
  }
  const int total = AnchorReduce(temp.reduce).Sum(local_missing);
  __syncthreads();
  if (threadIdx.x == 0) {
    missing_total = total;
    evicted = 0;
  }
  __syncthreads();
  if (missing_total == 0) {
    return;
  }

  for (int tile = 0; tile < keep - low_begin; tile += kThreads) {
    const int rank = keep - 1 - tile - threadIdx.x;
    const int id = rank >= low_begin
        ? sorted_ids[segment_offset + rank]
        : 0;
    const int candidate = rank >= low_begin && !anchors[id];
    int prefix = 0;
    int tile_total = 0;
    AnchorScan(temp.scan).ExclusiveSum(candidate, prefix, tile_total);
    __syncthreads();
    if (candidate && evicted + prefix < missing_total) {
      precision_map[segment_offset + id] = kSkip;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      evicted += tile_total;
    }
    __syncthreads();
    if (evicted >= missing_total) {
      break;
    }
  }
}

__global__ __launch_bounds__(kThreads) void pack_active_rows_kernel(
    const uint8_t* __restrict__ precision_map,
    int* __restrict__ packed_ids,
    int* __restrict__ low_counts,
    int* __restrict__ middle_counts,
    int* __restrict__ high_counts,
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

  PhaseCounts local{0, 0, 0};
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const uint8_t code = precision_map[row_offset + column];
    local.middle += code == kMiddle;
    local.high += code == kHigh;
    local.low += code == kLow;
    packed_ids[row_offset + column] = 0;
  }
  const PhaseCounts total = RowReduce(temp.reduce).Reduce(local, AddPhaseCounts{});
  __syncthreads();
  if (threadIdx.x == 0) {
    totals[0] = total.middle;
    totals[1] = total.high;
    totals[2] = total.low;
    running[0] = running[1] = running[2] = 0;
    low_counts[row] = total.low;
    middle_counts[row] = total.middle;
    high_counts[row] = total.high;
  }
  __syncthreads();

  for (int tile = 0; tile < columns; tile += kThreads) {
    const int column = tile + threadIdx.x;
    const uint8_t code =
        column < columns ? precision_map[row_offset + column] : kSkip;
    const bool valid = column < columns;
    const PhaseCounts is_phase{
        valid && code == kMiddle,
        valid && code == kHigh,
        valid && code == kLow};
    PhaseCounts prefix;
    PhaseCounts tile_counts;
    RowScan(temp.scan).ExclusiveScan(
        is_phase,
        prefix,
        PhaseCounts{0, 0, 0},
        AddPhaseCounts{},
        tile_counts);
    __syncthreads();

    if (is_phase.low) {
      packed_ids[row_offset + running[2] + prefix.low] = column;
    } else if (is_phase.middle) {
      packed_ids[row_offset + totals[2] + running[0] + prefix.middle] = column;
    } else if (is_phase.high) {
      packed_ids[
          row_offset + totals[2] + totals[0] + running[1] + prefix.high] =
          column;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      running[0] += tile_counts.middle;
      running[1] += tile_counts.high;
      running[2] += tile_counts.low;
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreads) void materialize_physical_route_kernel(
    const int* __restrict__ logical_ids,
    const int* __restrict__ low_counts,
    const int* __restrict__ middle_counts,
    const int* __restrict__ high_counts,
    int* __restrict__ physical_ids,
    int* __restrict__ physical_low_counts,
    int* __restrict__ physical_middle_counts,
    int* __restrict__ physical_high_counts,
    int row_count,
    int logical_columns,
    int physical_columns,
    int factor,
    int prefix_blocks,
    int prefix_phase,
    bool prefix_first,
    bool has_high) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int low = low_counts[row];
  const int middle = middle_counts[row];
  const int high = has_high ? high_counts[row] : 0;
  const int active = low + middle + high;
  const bool implicit_high_prefix =
      MPA_IMPLICIT_HIGH_PREFIX && prefix_phase == 2;
  const int prefix_position = prefix_first
      ? 0
      : (prefix_phase == 0
            ? 0
            : (prefix_phase == 1 ? low * factor : active * factor));

  for (int column = threadIdx.x; column < physical_columns;
       column += blockDim.x) {
    physical_ids[row * physical_columns + column] = 0;
  }
  __syncthreads();

  for (int position = threadIdx.x; position < active;
       position += blockDim.x) {
    int destination = position * factor;
    if (!implicit_high_prefix && destination >= prefix_position) {
      destination += prefix_blocks;
    }
    const int first =
        prefix_blocks + logical_ids[row * logical_columns + position] * factor;
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
      if (stage < factor) {
        physical_ids[row * physical_columns + destination + stage] =
            first + stage;
      }
    }
  }
  if (!implicit_high_prefix) {
    for (int prefix = threadIdx.x; prefix < prefix_blocks;
         prefix += blockDim.x) {
      physical_ids[row * physical_columns + prefix_position + prefix] = prefix;
    }
  }
  if (threadIdx.x == 0) {
    physical_low_counts[row] =
        low * factor + (prefix_phase == 0 ? prefix_blocks : 0);
    physical_middle_counts[row] =
        middle * factor + (prefix_phase == 1 ? prefix_blocks : 0);
    if (has_high) {
      physical_high_counts[row] =
          high * factor + (prefix_phase == 2 ? prefix_blocks : 0);
    }
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MPA_ROUTE_PRECISION_FUNCTION(
    torch::Tensor probability,
    int64_t n16_value,
    int64_t n8_value,
    int64_t n4_value,
    std::optional<torch::Tensor> anchors,
    std::optional<torch::Tensor> anchor_ids,
    int64_t anchor_count_value) {
  TORCH_CHECK(
      probability.defined() && probability.is_cuda() &&
          probability.is_contiguous() &&
          probability.scalar_type() == at::ScalarType::Half,
      "probability must be contiguous CUDA FP16");
  TORCH_CHECK(
      probability.dim() == 4 && probability.size(0) > 0 &&
          probability.size(1) > 0 && probability.size(2) > 0 &&
          probability.size(2) == probability.size(3),
      "probability must be square [B,H,R,R]");
  const int n16 = require_nonnegative_int32(n16_value, "n16");
  const int n8 = require_nonnegative_int32(n8_value, "n8");
  const int n4 = require_nonnegative_int32(n4_value, "n4");
  const int anchor_count =
      require_nonnegative_int32(anchor_count_value, "anchor_count");
  TORCH_CHECK(
      n16_value <= std::numeric_limits<int64_t>::max() - n8_value &&
          n16_value + n8_value <=
              std::numeric_limits<int64_t>::max() - n4_value,
      "n16+n8+n4 exceeds int64 range");
  const int64_t keep_value = n16_value + n8_value + n4_value;
  TORCH_CHECK(keep_value > 0, "retained count must be positive");

  const int64_t rows = probability.size(2);
  const int lowest_count = n4 ? n4 : (n8 ? n8 : n16);
  if (anchors.has_value()) {
    TORCH_CHECK(
        anchors->defined() && anchors->is_cuda() && anchors->is_contiguous() &&
            anchors->scalar_type() == at::ScalarType::Bool &&
            anchors->device() == probability.device() && anchors->dim() == 2 &&
            anchors->size(0) == rows && anchors->size(1) == rows,
        "anchors must be contiguous CUDA bool [R,R] on the probability device");
    TORCH_CHECK(
        anchor_ids.has_value() && anchor_ids->defined() &&
            anchor_ids->is_cuda() && anchor_ids->is_contiguous() &&
            anchor_ids->scalar_type() == at::ScalarType::Int &&
            anchor_ids->device() == probability.device() &&
            anchor_ids->dim() == 1 && anchor_ids->numel() == anchor_count,
        "anchor_ids must be contiguous CUDA int32 [anchor_count]");
    TORCH_CHECK(
        anchor_count <= lowest_count,
        "anchor_count exceeds the configured lowest-precision budget");
  } else {
    TORCH_CHECK(
        !anchor_ids.has_value() && anchor_count == 0,
        "anchor_ids must be absent and anchor_count zero when anchors are disabled");
  }
  const int64_t segment_items_value =
      checked_positive_product(rows, rows, "route items per head");
  const int64_t segments_value = checked_positive_product(
      probability.size(0), probability.size(1), "route segments");
  const int64_t total_items_value = checked_positive_product(
      segments_value, segment_items_value, "total route items");
  const int64_t row_count_value =
      checked_positive_product(segments_value, rows, "route rows");
  TORCH_CHECK(keep_value <= segment_items_value, "retained count exceeds R*R");
  TORCH_CHECK(
      segment_items_value <= std::numeric_limits<int>::max() &&
          total_items_value <= std::numeric_limits<int>::max() &&
          row_count_value <= std::numeric_limits<int>::max(),
      "route geometry exceeds CUB int32 range");
  TORCH_CHECK(segments_value <= kMaxSegments, "B*H exceeds global route key");

  c10::cuda::CUDAGuard device_guard(probability.device());
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(probability.get_device());
  TORCH_CHECK(
      MPA_ROUTE_DEVICE_OK(properties),
      "native H3 route requires ", MPA_ROUTE_DEVICE_LABEL);

  const int segment_items = static_cast<int>(segment_items_value);
  const int segments = static_cast<int>(segments_value);
  const int total_items = static_cast<int>(total_items_value);
  const int row_count = static_cast<int>(row_count_value);
  const int keep = static_cast<int>(keep_value);
  auto int_options = probability.options().dtype(at::ScalarType::Int);
  auto byte_options = probability.options().dtype(at::ScalarType::Byte);
  auto precision_map = torch::empty(probability.sizes(), byte_options);
  auto block_ids = torch::empty(probability.sizes(), int_options);
  auto sorted_ids = torch::empty_like(block_ids);
  auto input_keys = torch::empty(probability.sizes(), int_options);
  auto sorted_keys = torch::empty_like(input_keys);
  auto low_counts = torch::empty(
      {probability.size(0), probability.size(1), rows}, int_options);
  auto middle_counts = torch::empty_like(low_counts);
  auto high_counts = torch::empty_like(low_counts);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(probability.get_device());

  const int64_t blocks = (total_items_value + kThreads - 1) / kThreads;
  TORCH_CHECK(blocks <= kMaxGridX, "route grid.x exceeds CUDA limit");
  initialize_composite_sort_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(probability.data_ptr<at::Half>()),
      reinterpret_cast<uint32_t*>(input_keys.data_ptr<int>()),
      block_ids.data_ptr<int>(),
      total_items,
      segment_items);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int key_bits = kProbabilityBits + required_unsigned_bits(segments);
  size_t workspace_bytes = 0;
  cub::DoubleBuffer<uint32_t> query_keys(
      reinterpret_cast<uint32_t*>(input_keys.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(sorted_keys.data_ptr<int>()));
  cub::DoubleBuffer<int> query_values(
      block_ids.data_ptr<int>(), sorted_ids.data_ptr<int>());
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr,
      workspace_bytes,
      query_keys,
      query_values,
      total_items,
      0,
      key_bits,
      stream));
  TORCH_CHECK(
      workspace_bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "route workspace exceeds tensor range");
  auto workspace = torch::empty(
      {static_cast<int64_t>(workspace_bytes)}, byte_options);
  cub::DoubleBuffer<uint32_t> sort_keys(
      reinterpret_cast<uint32_t*>(input_keys.data_ptr<int>()),
      reinterpret_cast<uint32_t*>(sorted_keys.data_ptr<int>()));
  cub::DoubleBuffer<int> sort_values(
      block_ids.data_ptr<int>(), sorted_ids.data_ptr<int>());
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      workspace.data_ptr<uint8_t>(),
      workspace_bytes,
      sort_keys,
      sort_values,
      total_items,
      0,
      key_bits,
      stream));

  scatter_precision_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      sort_values.Current(),
      precision_map.data_ptr<uint8_t>(),
      total_items,
      segment_items,
      n16,
      n16 + n8,
      keep);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (anchors.has_value()) {
    const int low_begin = n4 ? n16 + n8 : (n8 ? n16 : 0);
    const uint8_t low_code = n4 ? kLow : (n8 ? kMiddle : kHigh);
    apply_anchor_budget_kernel<<<
        static_cast<unsigned int>(segments), kThreads, 0, stream>>>(
        sort_values.Current(),
        anchors->data_ptr<bool>(),
        anchor_ids->data_ptr<int>(),
        precision_map.data_ptr<uint8_t>(),
        anchor_count,
        segment_items,
        low_begin,
        keep,
        low_code);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  TORCH_CHECK(row_count_value <= kMaxGridX, "row-pack grid.x exceeds CUDA limit");
  pack_active_rows_kernel<<<
      static_cast<unsigned int>(row_count_value), kThreads, 0, stream>>>(
      precision_map.data_ptr<uint8_t>(),
      block_ids.data_ptr<int>(),
      low_counts.data_ptr<int>(),
      middle_counts.data_ptr<int>(),
      high_counts.data_ptr<int>(),
      row_count,
      static_cast<int>(rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {block_ids, low_counts, middle_counts, high_counts};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MPA_MATERIALIZE_ROUTE_FUNCTION(
    torch::Tensor logical_ids,
    torch::Tensor low_counts,
    torch::Tensor middle_counts,
    torch::Tensor high_counts,
    int64_t query_block_size_value,
    int64_t prefix_blocks_value,
    int64_t prefix_phase_value,
    bool prefix_first,
    bool has_high) {
  TORCH_CHECK(
      logical_ids.defined() && logical_ids.is_cuda() &&
          logical_ids.is_contiguous() &&
          logical_ids.scalar_type() == at::ScalarType::Int &&
          logical_ids.dim() == 4,
      "logical_ids must be contiguous CUDA int32 [B,H,R,C]");
  const auto check_counts = [&](const torch::Tensor& counts, const char* name) {
    TORCH_CHECK(
        counts.defined() && counts.is_cuda() && counts.is_contiguous() &&
            counts.scalar_type() == at::ScalarType::Int &&
            counts.device() == logical_ids.device() && counts.dim() == 3 &&
            counts.sizes() == logical_ids.sizes().slice(0, 3),
        name, " must be contiguous CUDA int32 [B,H,R]");
  };
  check_counts(low_counts, "low_counts");
  check_counts(middle_counts, "middle_counts");
  check_counts(high_counts, "high_counts");
  TORCH_CHECK(
      query_block_size_value == 64 || query_block_size_value == 128,
      "query_block_size must be 64 or 128");
  const int prefix_blocks =
      require_nonnegative_int32(prefix_blocks_value, "prefix_blocks");
  const int prefix_phase =
      require_nonnegative_int32(prefix_phase_value, "prefix_phase");
  TORCH_CHECK(prefix_phase < 3, "prefix_phase must be 0, 1, or 2");
  TORCH_CHECK(
      has_high || prefix_phase != 2,
      "FP16 prefix requires active high counts");

  const int factor = static_cast<int>(query_block_size_value / 64);
  const int logical_columns = static_cast<int>(logical_ids.size(3));
  const int64_t physical_columns_value =
      checked_positive_product(logical_columns, factor, "physical route columns") +
      prefix_blocks;
  TORCH_CHECK(
      physical_columns_value <= std::numeric_limits<int>::max(),
      "physical route columns exceed int32 range");
  const int64_t row_count_value = logical_ids.numel() / logical_columns;
  TORCH_CHECK(
      row_count_value > 0 && row_count_value <= kMaxGridX &&
          row_count_value <= std::numeric_limits<int>::max(),
      "physical route row count exceeds CUDA range");

  c10::cuda::CUDAGuard device_guard(logical_ids.device());
  auto physical_sizes = logical_ids.sizes().vec();
  physical_sizes.back() = physical_columns_value;
  auto physical_ids = torch::empty(physical_sizes, logical_ids.options());
  auto physical_low_counts = torch::empty_like(low_counts);
  auto physical_middle_counts = torch::empty_like(middle_counts);
  auto physical_high_counts = has_high
      ? torch::empty_like(high_counts)
      : torch::empty({0}, high_counts.options());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(logical_ids.get_device());
  materialize_physical_route_kernel<<<
      static_cast<unsigned int>(row_count_value), kThreads, 0, stream>>>(
      logical_ids.data_ptr<int>(),
      low_counts.data_ptr<int>(),
      middle_counts.data_ptr<int>(),
      high_counts.data_ptr<int>(),
      physical_ids.data_ptr<int>(),
      physical_low_counts.data_ptr<int>(),
      physical_middle_counts.data_ptr<int>(),
      has_high ? physical_high_counts.data_ptr<int>() : nullptr,
      static_cast<int>(row_count_value),
      logical_columns,
      static_cast<int>(physical_columns_value),
      factor,
      prefix_blocks,
      prefix_phase,
      prefix_first,
      has_high);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      physical_ids,
      physical_low_counts,
      physical_middle_counts,
      physical_high_counts};
}

#undef MPA_ROUTE_PRECISION_FUNCTION
#undef MPA_MATERIALIZE_ROUTE_FUNCTION
#undef MPA_ROUTE_DEVICE_OK
#undef MPA_ROUTE_DEVICE_LABEL
#undef MPA_ROUTE_THREADS
#undef MPA_IMPLICIT_HIGH_PREFIX
