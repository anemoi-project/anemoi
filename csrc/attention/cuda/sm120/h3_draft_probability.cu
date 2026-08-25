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
 * Ported from mixed_precision_attention commit
 * e318368c15a2962885df2117c35710e86837d5d1.  FP16 pooled operands feed a
 * cuBLAS GEMM with FP32 accumulation; row softmax keeps max, sum, and exp in
 * FP32 and rounds only the final probability to FP16.
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

#include "api.h"

namespace {

constexpr int kSoftmaxThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kSoftmaxWarps = kSoftmaxThreads / kWarpSize;
constexpr int64_t kMaxGridX = 2147483647LL;
constexpr cublasComputeType_t kDraftComputeType = CUBLAS_COMPUTE_32F;
constexpr cublasGemmAlgo_t kDraftAlgorithm = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

inline int64_t checked_positive_product(
    int64_t lhs, int64_t rhs, const char* description) {
  TORCH_CHECK(lhs > 0 && rhs > 0, description, " factors must be positive");
  TORCH_CHECK(
      lhs <= std::numeric_limits<int64_t>::max() / rhs,
      description,
      " exceeds int64 range");
  return lhs * rhs;
}

inline void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      operation,
      " failed with cuBLAS status ",
      static_cast<int>(status));
}

inline void check_pool_tensor(
    const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be a defined tensor");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous BHSD");
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Half,
      name,
      " must have dtype torch.float16");
  TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B,H,R,D]");
  TORCH_CHECK(
      tensor.size(0) > 0 && tensor.size(1) > 0 && tensor.size(2) > 0,
      name,
      " batch, head, and row dimensions must be positive");
  TORCH_CHECK(
      tensor.size(3) == 64 || tensor.size(3) == 128,
      name,
      " head dimension must be 64 or 128");
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
  const float inverse_sum =
      1.0f / block_reduce<false>(local_sum, warp_scratch);

  for (int64_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const float normalized = expf(
        __half2float(logits_probability[row_offset + column]) - row_max) *
        inverse_sum;
    logits_probability[row_offset + column] = __float2half_rn(normalized);
  }
}

void launch_draft_gemm(
    const torch::Tensor& q_pool,
    const torch::Tensor& k_pool,
    torch::Tensor& logits,
    cublasHandle_t handle) {
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

  // Row-major Q*K^T is emitted through the equivalent column-major K*Q^T.
  // Choose the cheaper affine GQA batching axis exactly as the donor does.
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
    return;
  }

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

}  // namespace

torch::Tensor sm120_h3_draft_probability(
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
      properties->major == 12 && properties->minor == 0,
      "sm120_h3_draft_probability requires SM120, found sm_",
      properties->major,
      properties->minor);

  const int64_t rows = q_pool.size(2);
  const int64_t output_head_stride = checked_positive_product(
      rows, rows, "Draft probability head stride");
  const int64_t output_heads = checked_positive_product(
      q_pool.size(0), q_pool.size(1), "Draft B*Hq");
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

  launch_draft_gemm(q_pool, k_pool, logits, handle);
  const int64_t softmax_rows = checked_positive_product(
      output_heads, rows, "Draft softmax row count");
  TORCH_CHECK(
      softmax_rows <= kMaxGridX, "Draft softmax grid.x exceeds CUDA limit");
  row_softmax_fp16_kernel<<<
      static_cast<unsigned int>(softmax_rows),
      kSoftmaxThreads,
      0,
      stream>>>(
      reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
      softmax_rows,
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}
