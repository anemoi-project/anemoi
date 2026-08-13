/*
 * Copyright (c) 2025 by SpargeAttn team.
 * Copyright (c) 2026 mixed-attention project contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This project-owned SM89 derivative starts from the locked SpargeAttention
 * qk_int_sv_f8_block_sparse_attn_kernel (commit
 * ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a).  It retains Sparge/Sage's
 * shared-memory wrappers, INT8 QK path, online softmax, E4M3 P/V path and
 * stage-local FP16 PV accumulator.  The project adds homogeneous precision
 * lists, the raw-FP16 rescue phase, and the in-register precision boundary.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "primitives/cp_async.cuh"
#include "primitives/math.cuh"
#include "primitives/mma.cuh"
#include "primitives/permuted_smem.cuh"
#include "primitives/qattn/attn_utils.cuh"
#include "attention_decl.cuh"

#if !defined(MPA_ATTENTION_KERNEL_ENTRY) || !defined(MPA_ATTENTION_LAUNCH_ENTRY)
#if defined(MPA_PACKED_RASTER_MODE)
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm89_packed_raster_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm89_packed_raster
#else
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm89_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm89
#endif
#endif

namespace mpa::attention {

#ifndef MPA_CTA_Q
#define MPA_CTA_Q 128
#endif
constexpr uint32_t kCtaQ = MPA_CTA_Q;
static_assert(kCtaQ == 64 || kCtaQ == 128, "Q CTA must be 64 or 128");
constexpr uint32_t kCtaK = 64;
#ifndef MPA_FP16_CTA_K
#define MPA_FP16_CTA_K 64
#endif
constexpr uint32_t kFp16CtaK = MPA_FP16_CTA_K;
static_assert(
    kFp16CtaK == 32 || kFp16CtaK == 64,
    "FP16 stage K must be 32 or 64");
constexpr uint32_t kFp16StagesPerRasterPatch = kCtaQ / kFp16CtaK;
#ifndef MPA_FP16_PIPELINE
#define MPA_FP16_PIPELINE 0
#endif
constexpr bool kFp16Pipeline = MPA_FP16_PIPELINE != 0;
static_assert(!kFp16Pipeline || kFp16CtaK == 32);
#ifndef MPA_WARP_Q
#define MPA_WARP_Q 32
#endif
constexpr uint32_t kWarpQ = MPA_WARP_Q;
static_assert(
    kWarpQ == 16 || kWarpQ == 32,
    "each physical warp must own 16 or 32 query rows");
constexpr uint32_t kWarpK = 64;
constexpr uint32_t kPackInt8 = 16;
constexpr uint32_t kPackFp8 = 16;
constexpr uint32_t kPackHalf = 8;
constexpr uint32_t kMmaQkM = 16;
constexpr uint32_t kMmaQkN = 16;
constexpr uint32_t kMmaInt8K = 32;
constexpr uint32_t kMmaHalfK = 16;
constexpr uint32_t kMmaPvN = 16;

template <uint32_t HeadDim>
using LowQKSmem = smem_t<
    (HeadDim == 64 ? SwizzleMode::k64B : SwizzleMode::k128B),
    HeadDim / kPackInt8>;

using LowVSmem = smem_t<SwizzleMode::k64B, kCtaK / kPackFp8>;

template <uint32_t HeadDim>
using HalfQKVSmem = smem_t<SwizzleMode::k128B, HeadDim / kPackHalf>;

template <uint32_t HeadDim>
using OutputSmem = smem_t<SwizzleMode::k128B, HeadDim / kPackHalf>;

template <uint32_t NumTilesQ, uint32_t NumTilesV>
__device__ __forceinline__ void scale_ro_by_v_scale(
    float (&ro)[NumTilesQ][NumTilesV][8],
    const float* v_scale_base,
    float common_scale) {
  const uint32_t lane_id = get_lane_id();
  float scales[4];
#pragma unroll
  for (uint32_t fv = 0; fv < NumTilesV; ++fv) {
    reinterpret_cast<float2*>(scales)[0] =
        *reinterpret_cast<const float2*>(
            v_scale_base + fv * 16 + (lane_id % 4) * 2);
    reinterpret_cast<float2*>(scales)[1] =
        *reinterpret_cast<const float2*>(
            v_scale_base + fv * 16 + 8 + (lane_id % 4) * 2);
#pragma unroll
    for (uint32_t index = 0; index < 4; ++index) {
      scales[index] *= common_scale;
    }
#pragma unroll
    for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
      ro[fq][fv][0] *= scales[0];
      ro[fq][fv][1] *= scales[1];
      ro[fq][fv][2] *= scales[0];
      ro[fq][fv][3] *= scales[1];
      ro[fq][fv][4] *= scales[2];
      ro[fq][fv][5] *= scales[3];
      ro[fq][fv][6] *= scales[2];
      ro[fq][fv][7] *= scales[3];
    }
  }
}

template <uint32_t NumTilesQ, uint32_t NumTilesV>
__device__ __forceinline__ void leave_fp8_probability_and_v_domains(
    float (&ro)[NumTilesQ][NumTilesV][8],
    float (&d)[NumTilesQ][2],
    const float* v_scale_base) {
  scale_ro_by_v_scale(ro, v_scale_base, S_FP8_OFFSET_EXP_INV);
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      d[fq][row] *= S_FP8_OFFSET_EXP_INV;
    }
  }
}

template <uint32_t NumTilesQ, uint32_t NumTilesV>
__device__ __forceinline__ void normalize_d_inplace(
    float ro[][NumTilesV][8],
    float d[][2]) {
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      d[fq][row] += __shfl_xor_sync(0xffffffff, d[fq][row], 1);
      d[fq][row] += __shfl_xor_sync(0xffffffff, d[fq][row], 2);
      d[fq][row] = math::ptx_rcp(d[fq][row]);
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < NumTilesV; ++fv) {
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        ro[fq][fv][element] *= d[fq][(element % 4) / 2];
      }
    }
  }
}

template <uint32_t HeadDim, uint32_t NumTilesQ>
__device__ __forceinline__ void restore_smoothed_low_rowmax(
    const HalfQKVSmem<HeadDim>& smem_q,
    const half* k_mean,
    float (&m)[NumTilesQ][2],
    uint32_t query_block,
    uint32_t qo_len,
    float base2_softmax_scale) {
  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_q = get_warp_idx_q<kCtaQ / kWarpQ, kCtaK / kWarpK>();
  const uint32_t lane_in_row = lane_id % 4;

#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t row_half = 0; row_half < 2; ++row_half) {
      const uint32_t local_row =
          warp_q * kWarpQ + fq * kMmaQkM + lane_id / 4 + row_half * 8;
      const uint32_t global_row = query_block * kCtaQ + local_row;
      float dot = 0.0f;
      const bool valid_row = global_row < qo_len;
      if (valid_row) {
#pragma unroll
        for (uint32_t pack = lane_in_row; pack < HeadDim / kPackHalf; pack += 4) {
          const b128_t q_pack =
              smem_q.base[smem_q.get_permuted_offset(local_row, pack)];
          const b128_t mean_pack = reinterpret_cast<const b128_t*>(k_mean)[pack];
          const half* q_values = reinterpret_cast<const half*>(&q_pack);
          const half* mean_values = reinterpret_cast<const half*>(&mean_pack);
#pragma unroll
          for (uint32_t element = 0; element < kPackHalf; ++element) {
            dot = fmaf(
                __half2float(q_values[element]),
                __half2float(mean_values[element]),
                dot);
          }
        }
      }
      // Every lane in the warp must execute both shuffles.  Query-tail
      // predicates vary between four-lane row groups, so placing these under
      // valid_row while using a full mask would be undefined.
      dot += __shfl_xor_sync(0xffffffff, dot, 1);
      dot += __shfl_xor_sync(0xffffffff, dot, 2);
      if (valid_row) {
        m[fq][row_half] = fmaf(dot, base2_softmax_scale, m[fq][row_half]);
      }
    }
  }
}

union Half2Bits {
  half2 value;
  uint32_t bits;
};

template <uint32_t NumTilesQ, uint32_t NumTilesK>
__device__ __forceinline__ void pack_rs_fp16_inplace(
    float rs[][NumTilesK][8]) {
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
      // Materialize all four words before overwriting the first half of the
      // FP32 fragment.  The second half then dies and the first half becomes
      // the packed FP16 operand without a second 32-register D128 array.
      Half2Bits packed[4];
#pragma unroll
      for (uint32_t pair = 0; pair < 4; ++pair) {
        packed[pair].value = __float22half2_rn(
            reinterpret_cast<float2*>(rs[fq][fk])[pair]);
      }
#pragma unroll
      for (uint32_t pair = 0; pair < 4; ++pair) {
        rs[fq][fk][pair] = __uint_as_float(packed[pair].bits);
      }
    }
  }
}

// The donor helper keeps every V-tile FP16 stage accumulator live at once.
// That is ideal for instruction scheduling in a homogeneous kernel, but the
// D128 mixed kernel already carries 128 FP32 output values per lane and would
// spill.  Process one compile-time V tile at a time instead.  Each tile still
// accumulates all K=64 in FP16 and is promoted exactly once at the physical
// stage boundary, preserving the mixed-precision accumulator contract.
template <
    uint32_t Fv, uint32_t NumTilesQ, uint32_t NumTilesK,
    uint32_t NumTilesV, SwizzleMode Swizzle, uint32_t Stride,
    bool SplitN8 = false>
__device__ __forceinline__ void compute_fp16_sv_stage_tilewise(
    const smem_t<Swizzle, Stride>& smem_v,
    float rs_packed[][NumTilesK][8],
    float ro[][NumTilesV][8]) {
  if constexpr (Fv < NumTilesV) {
    {
      const uint32_t lane_id = get_lane_id();
      if constexpr (SplitN8) {
        static_assert(NumTilesQ == 1, "m16n8 split requires one Q tile");
        {
          uint32_t ro_half[2];

#pragma unroll
          for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
            uint32_t rv[2];
            const uint32_t v_offset = smem_v.get_permuted_offset(
                (lane_id & 15U) + fk * kMmaHalfK, Fv * 2);
            b128_t* smem_ptr = smem_v.base + v_offset;
            const uint32_t smem_int_ptr =
                static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                "{%0, %1}, [%2];\n"
                : "=r"(rv[0]), "=r"(rv[1])
                : "r"(smem_int_ptr));
            uint32_t rs_fragment[4];
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
              rs_fragment[word] = __float_as_uint(rs_packed[0][fk][word]);
            }
            if (fk == 0) {
              mma::mma_sync_m16n8k16_row_col_f16f16f16<
                  mma::MMAMode::kInit>(ro_half, rs_fragment, rv);
            } else {
              mma::mma_sync_m16n8k16_row_col_f16f16f16<
                  mma::MMAMode::kInplaceUpdate>(
                  ro_half, rs_fragment, rv);
            }
          }

#pragma unroll
          for (uint32_t pair = 0; pair < 2; ++pair) {
            const half2 value =
                reinterpret_cast<const half2*>(ro_half)[pair];
            ro[0][Fv][pair * 2] += __half2float(value.x);
            ro[0][Fv][pair * 2 + 1] += __half2float(value.y);
          }
        }

        {
          uint32_t ro_half[2];

#pragma unroll
          for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
            uint32_t rv[2];
            const uint32_t v_offset = smem_v.get_permuted_offset(
                (lane_id & 15U) + fk * kMmaHalfK, Fv * 2 + 1);
            b128_t* smem_ptr = smem_v.base + v_offset;
            const uint32_t smem_int_ptr =
                static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                "{%0, %1}, [%2];\n"
                : "=r"(rv[0]), "=r"(rv[1])
                : "r"(smem_int_ptr));
            uint32_t rs_fragment[4];
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
              rs_fragment[word] = __float_as_uint(rs_packed[0][fk][word]);
            }
            if (fk == 0) {
              mma::mma_sync_m16n8k16_row_col_f16f16f16<
                  mma::MMAMode::kInit>(ro_half, rs_fragment, rv);
            } else {
              mma::mma_sync_m16n8k16_row_col_f16f16f16<
                  mma::MMAMode::kInplaceUpdate>(
                  ro_half, rs_fragment, rv);
            }
          }

#pragma unroll
          for (uint32_t pair = 0; pair < 2; ++pair) {
            const half2 value =
                reinterpret_cast<const half2*>(ro_half)[pair];
            ro[0][Fv][4 + pair * 2] += __half2float(value.x);
            ro[0][Fv][5 + pair * 2] += __half2float(value.y);
          }
        }
      } else {
        uint32_t ro_stage[NumTilesQ][4];

#pragma unroll
        for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
          uint32_t rv[4];
          const uint32_t v_offset = smem_v.get_permuted_offset(
              lane_id % 16 + fk * kMmaHalfK,
              lane_id / 16 + Fv * (kMmaPvN / kPackHalf));
          smem_v.ldmatrix_m8n8x4_trans(v_offset, rv);
#pragma unroll
          for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
            uint32_t rs_fragment[4];
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
              rs_fragment[word] = __float_as_uint(rs_packed[fq][fk][word]);
            }
            if (fk == 0) {
              mma::mma_sync_m16n16k16_row_col_f16f16f16<
                  mma::MMAMode::kInit>(ro_stage[fq], rs_fragment, rv);
            } else {
              mma::mma_sync_m16n16k16_row_col_f16f16f16<
                  mma::MMAMode::kInplaceUpdate>(
                  ro_stage[fq], rs_fragment, rv);
            }
          }
        }

#pragma unroll
        for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
          for (uint32_t pair = 0; pair < 4; ++pair) {
            const half2 value =
                reinterpret_cast<const half2*>(ro_stage[fq])[pair];
            ro[fq][Fv][pair * 2] += __half2float(value.x);
            ro[fq][Fv][pair * 2 + 1] += __half2float(value.y);
          }
        }
      }
    }
    compute_fp16_sv_stage_tilewise<
        Fv + 1, NumTilesQ, NumTilesK, NumTilesV, Swizzle, Stride,
        SplitN8>(
        smem_v, rs_packed, ro);
  }
}

// D128 mixed kernels also need to lower the donor FP8 PV peak.  The all-FP8
// family keeps the donor helper unchanged; only the compile-time mixed family
// changes loop order so one V tile's stage accumulator is live at once. CUDA
// 12.8 added FP8-input/FP16-accumulator MMA; CUDA 12.4-12.7 use the equivalent
// FP32-accumulator instruction instead of compiling the unsupported path to a
// device breakpoint.
template <
    uint32_t Fv, uint32_t NumTilesQ, uint32_t NumTilesK,
    uint32_t NumTilesV, SwizzleMode Swizzle, uint32_t Stride>
__device__ __forceinline__ void compute_fp8_sv_stage_tilewise(
    const smem_t<Swizzle, Stride>& smem_v,
    uint32_t rs_fp8[][NumTilesK / 2][4],
    float ro[][NumTilesV][8]) {
  if constexpr (Fv < NumTilesV) {
    {
#ifdef MMA_F8F8F16_M16N8K16_ENABLED
      uint32_t ro_stage[NumTilesQ][4];
#else
      float ro_stage[NumTilesQ][8];
#endif
      const uint32_t lane_id = get_lane_id();
      const uint32_t row_base = lane_id % 8 + (lane_id / 16) * 8;
      const uint32_t col_base = (lane_id / 8) % 2;

#pragma unroll
      for (uint32_t fk = 0; fk < NumTilesK / 2; ++fk) {
        uint32_t rv[4];
        const uint32_t v_offset = smem_v.get_permuted_offset(
            row_base + Fv * kMmaPvN, col_base + fk * 2);
        smem_v.ldmatrix_m8n8x4(v_offset, rv);
#pragma unroll
        for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#ifdef MMA_F8F8F16_M16N8K16_ENABLED
          if (fk == 0) {
            mma::mma_sync_m16n16k32_row_col_f8f8f16<
                mma::MMAMode::kInit>(ro_stage[fq], rs_fp8[fq][fk], rv);
          } else {
            mma::mma_sync_m16n16k32_row_col_f8f8f16<
                mma::MMAMode::kInplaceUpdate>(
                ro_stage[fq], rs_fp8[fq][fk], rv);
          }
#else
          if (fk == 0) {
            mma::mma_sync_m16n16k32_row_col_f8f8f32<
                mma::MMAMode::kInit>(
                ro_stage[fq], rs_fp8[fq][fk], rv);
          } else {
            mma::mma_sync_m16n16k32_row_col_f8f8f32<
                mma::MMAMode::kInplaceUpdate>(
                ro_stage[fq], rs_fp8[fq][fk], rv);
          }
#endif
        }
      }

#pragma unroll
      for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#ifdef MMA_F8F8F16_M16N8K16_ENABLED
#pragma unroll
        for (uint32_t pair = 0; pair < 4; ++pair) {
          float unpacked[2];
          unpack_half2_from_uint32_to_float(unpacked, ro_stage[fq][pair]);
          ro[fq][Fv][pair * 2] += unpacked[0];
          ro[fq][Fv][pair * 2 + 1] += unpacked[1];
        }
#else
#pragma unroll
        for (uint32_t element = 0; element < 8; ++element) {
          ro[fq][Fv][element] += ro_stage[fq][element];
        }
#endif
      }
    }
    compute_fp8_sv_stage_tilewise<
        Fv + 1, NumTilesQ, NumTilesK, NumTilesV, Swizzle, Stride>(
        smem_v, rs_fp8, ro);
  }
}

// D128 uses the already-consumed fq=0 half of each warp's Q shared-memory
// slice as a transient 16-KiB score stash.  Only those 64 Q rows need to be
// restored before a later FP16 physical stage; fq=1 remains resident.
template <bool Drain = true>
__device__ __forceinline__ void reload_d128_fq0_q(
    const HalfQKVSmem<128>& smem_q,
    const half* q16,
    uint32_t batch_id,
    uint32_t head_id,
    uint32_t num_qo_heads,
    uint32_t query_block,
    uint32_t qo_len) {
  constexpr uint32_t packs_per_row = 128 / kPackHalf;
  constexpr uint32_t rows_per_warp = kWarpQ / 2;
  constexpr uint32_t compact_rows =
      (kCtaQ / kWarpQ) * rows_per_warp;
  constexpr uint32_t total_packs = compact_rows * packs_per_row;
  constexpr uint32_t cta_threads =
      32 * (kCtaQ / kWarpQ) * (kCtaK / kWarpK);
  const uint32_t linear_thread = get_warp_id() * 32 + get_lane_id();

#pragma unroll 1
  for (uint32_t copy = 0; copy < total_packs / cta_threads; ++copy) {
    const uint32_t compact_index = linear_thread + copy * cta_threads;
    const uint32_t compact_row = compact_index / packs_per_row;
    const uint32_t pack = compact_index % packs_per_row;
    const uint32_t local_row =
        (compact_row / rows_per_warp) * kWarpQ +
        compact_row % rows_per_warp;
    const uint32_t global_row = query_block * kCtaQ + local_row;
    const half* source =
        q16 + ((batch_id * num_qo_heads + head_id) * qo_len + global_row) *
                  128 +
        pack * kPackHalf;
    smem_q.load_128b_async<cp_async::SharedMemFillMode::kFillZero>(
        smem_q.get_permuted_offset(local_row, pack),
        source,
        global_row < qo_len);
  }
  cp_async::commit_group();
  if constexpr (Drain) {
    cp_async::wait_group<0>();
    __syncthreads();
  }
}

template <uint32_t NumTilesK>
__device__ __forceinline__ void stash_d128_fq0_scores(
    int8_t* smem,
    float scores[][NumTilesK][8]) {
  constexpr uint32_t score_values = NumTilesK * 8;
  constexpr uint32_t score_packs = score_values / 4;
  constexpr uint32_t warp_lanes = 32;
  constexpr uint32_t q_slice_values =
      kWarpQ * 128 * sizeof(half) / sizeof(float);
  constexpr uint32_t q_slice_packs = q_slice_values / 4;
  static_assert(score_packs * warp_lanes <= q_slice_packs);
  float4* warp_slots = reinterpret_cast<float4*>(smem) +
      get_warp_id() * q_slice_packs;
  const float4* score_frag = reinterpret_cast<float4*>(scores[0][0]);
#pragma unroll
  for (uint32_t pack = 0; pack < score_packs; ++pack) {
    // Keep the 16-byte vector access, but make lanes adjacent within each
    // pack.  A lane-major score array strides every lane by 128 bytes and
    // maps the entire warp to the same four shared-memory banks.
    warp_slots[pack * warp_lanes + get_lane_id()] = score_frag[pack];
  }
}

template <uint32_t NumTilesK>
__device__ __forceinline__ void load_d128_fq0_scores(
    const int8_t* smem,
    float scores[][NumTilesK][8]) {
  constexpr uint32_t score_values = NumTilesK * 8;
  constexpr uint32_t score_packs = score_values / 4;
  constexpr uint32_t warp_lanes = 32;
  constexpr uint32_t q_slice_values =
      kWarpQ * 128 * sizeof(half) / sizeof(float);
  constexpr uint32_t q_slice_packs = q_slice_values / 4;
  static_assert(score_packs * warp_lanes <= q_slice_packs);
  const float4* warp_slots = reinterpret_cast<const float4*>(smem) +
      get_warp_id() * q_slice_packs;
  float4* score_frag = reinterpret_cast<float4*>(scores[0][0]);
#pragma unroll
  for (uint32_t pack = 0; pack < score_packs; ++pack) {
    score_frag[pack] = warp_slots[pack * warp_lanes + get_lane_id()];
  }
}

template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
__global__ __launch_bounds__(
    32 * (kCtaQ / kWarpQ) * (kCtaK / kWarpK))
void MPA_ATTENTION_KERNEL_ENTRY(
    int8_t* __restrict__ q8,
    int8_t* __restrict__ k8,
    __nv_fp8_e4m3* __restrict__ v8,
    half* __restrict__ q16,
    half* __restrict__ k16,
    half* __restrict__ v16,
    half* __restrict__ k_mean,
    half* __restrict__ output,
    int32_t* __restrict__ fp8_lut,
    int32_t* __restrict__ fp8_count,
    int32_t* __restrict__ fp16_lut,
    int32_t* __restrict__ fp16_count,
    float* __restrict__ q_scale,
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
#if defined(MPA_K64_BLOCK_MODE)
    const int32_t* __restrict__ valid_k_counts,
    float* __restrict__ lse,
    uint32_t fp16_prefix_stages,
#endif
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t padded_kv_len,
    uint32_t num_kv_groups,
    float softmax_scale) {
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(HasFp8 || HasFp16);
  static_assert(!SmoothK || (HasFp8 && HasFp16));

  constexpr uint32_t num_warps_q = kCtaQ / kWarpQ;
  constexpr uint32_t num_warps_k = kCtaK / kWarpK;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = kWarpQ / kMmaQkM;
  constexpr uint32_t num_tiles_k = kWarpK / kMmaQkN;
  constexpr uint32_t num_tiles_v = HeadDim / kMmaPvN;
  constexpr uint32_t low_qk_inner = HeadDim / kMmaInt8K;
  constexpr uint32_t half_qk_inner = HeadDim / kMmaHalfK;

  extern __shared__ int8_t smem[];
  uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();
  const uint32_t batch_id = blockIdx.z;
  const uint32_t query_block = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t kv_head = head_id / num_kv_groups;
  const uint32_t num_kv_heads = num_qo_heads / num_kv_groups;
  const uint32_t num_query_blocks = gridDim.x;
  const uint32_t num_physical_stages = div_ceil(kv_len, kCtaK);
  const uint32_t metadata_row =
      (batch_id * num_qo_heads + head_id) * num_query_blocks + query_block;

  const uint32_t low_iterations = HasFp8 ? fp8_count[metadata_row] : 0;
  const uint32_t high_iterations = HasFp16 ? fp16_count[metadata_row] : 0;
  if (low_iterations == 0 && high_iterations == 0) {
    return;
  }

  const float base2_softmax_scale = softmax_scale * math::log2e;

  float ro[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2];
  float d[num_tiles_q][2];
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; ++fv) {
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        ro[fq][fv][element] = 0.0f;
      }
    }
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      m[fq][row] = -5000000.0f;
      d[fq][row] = 1.0f;
    }
  }

  if constexpr (HasFp8) {
    if (low_iterations != 0) {
      using QKSmem = LowQKSmem<HeadDim>;
      QKSmem smem_q8(smem);
      QKSmem smem_k8(smem + kCtaQ * HeadDim);
      LowVSmem smem_v8(smem + (kCtaQ + kCtaK) * HeadDim);

      constexpr uint32_t qk_line_lanes = HeadDim == 64 ? 4 : 8;
      constexpr uint32_t qk_lines_per_warp = 32 / qk_line_lanes;
      constexpr uint32_t qk_smem_iters_row =
          HeadDim / (qk_line_lanes * kPackInt8);
      constexpr uint32_t q_smem_iters_col =
          kCtaQ / (num_warps * qk_lines_per_warp);
      constexpr uint32_t k_smem_iters_col =
          kCtaK / (num_warps * qk_lines_per_warp);
      constexpr uint32_t v_line_lanes = 4;
      constexpr uint32_t v_lines_per_warp = 8;
      constexpr uint32_t v_smem_iters_row =
          kCtaK / (v_line_lanes * kPackFp8);
      constexpr uint32_t v_smem_iters_col =
          HeadDim / (num_warps * v_lines_per_warp);

      int8_t* q_lane =
          q8 + ((batch_id * num_qo_heads + head_id) * qo_len +
                query_block * kCtaQ +
                kCtaQ / num_warps * warp_id + lane_id / qk_line_lanes) *
                   HeadDim +
          (lane_id % qk_line_lanes) * kPackInt8;
      uint32_t q_smem_load = smem_q8.get_permuted_offset(
          warp_id * qk_lines_per_warp * q_smem_iters_col +
              lane_id / qk_line_lanes,
          lane_id % qk_line_lanes);
      const uint32_t q_load_row =
          query_block * kCtaQ + kCtaQ / num_warps * warp_id +
          lane_id / qk_line_lanes;
      load_global_to_share<
          qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
          q_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                           : SwizzleMode::k128B),
          HeadDim / kPackInt8, kCtaQ>(
          q_lane, q_smem_load, HeadDim, smem_q8, q_load_row, qo_len);
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();

      const uint32_t q_smem_mma = smem_q8.get_permuted_offset(
          get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ + lane_id % 16,
          lane_id / 16);
      const uint32_t k_smem_mma = smem_k8.get_permuted_offset(
          lane_id % 8 + (lane_id / 16) * 8,
          (lane_id / 8) % 2);
      const float q_dequant_scale =
          q_scale[(batch_id * num_qo_heads + head_id) * num_query_blocks +
                  query_block];
      int32_t* low_lut = fp8_lut + metadata_row * num_physical_stages;
      uint32_t absolute_stage = 0;

      if constexpr (HasFp8 && !HasFp16) {
        // Keep the inherited Sparge/Sage all-FP8 copy/compute pipeline in the
        // same specialization.  K and V occupy disjoint shared-memory regions:
        // with two committed groups outstanding, wait_group<1> exposes the
        // older operand while the newer operand continues to overlap compute.
        int32_t* low_delta = low_lut;
        const float* k_scale_base =
            k_scale + (batch_id * num_kv_heads + kv_head) *
                          num_physical_stages;

        // Prologue: Q is already resident.  Issue predicated K0 followed by
        // padded V0, leaving both groups outstanding.
        absolute_stage += static_cast<uint32_t>(*low_delta++);
        int8_t* k_lane =
            k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                  absolute_stage * kCtaK +
                  kCtaK / num_warps * warp_id + lane_id / qk_line_lanes) *
                     HeadDim +
            (lane_id % qk_line_lanes) * kPackInt8;
        uint32_t k_smem_load = smem_k8.get_permuted_offset(
            warp_id * qk_lines_per_warp * k_smem_iters_col +
                lane_id / qk_line_lanes,
            lane_id % qk_line_lanes);
        uint32_t k_load_row =
            absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
            lane_id / qk_line_lanes;
        load_global_to_share<
            qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
            k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                             : SwizzleMode::k128B),
            HeadDim / kPackInt8, kCtaK>(
            k_lane, k_smem_load, HeadDim, smem_k8, k_load_row, kv_len);
        cp_async::commit_group();

        float current_dequant_scale =
            q_dequant_scale * k_scale_base[absolute_stage];

        int8_t* v_lane =
            reinterpret_cast<int8_t*>(v8) +
            ((batch_id * num_kv_heads + kv_head) * HeadDim +
             HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                padded_kv_len +
            absolute_stage * kCtaK +
            (lane_id % v_line_lanes) * kPackFp8;
        uint32_t v_smem_load = smem_v8.get_permuted_offset(
            warp_id * v_lines_per_warp * v_smem_iters_col +
                lane_id / v_line_lanes,
            lane_id % v_line_lanes);
        load_fp8_V_global_to_share<
            v_line_lanes, v_lines_per_warp, v_smem_iters_row,
            v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
            kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
        cp_async::commit_group();

        // Positions [0, n-3] retain donor rounding: the integer score is
        // converted to float without pre-dequantization and q*k is folded
        // into the softmax scale.
        for (uint32_t iteration = 0;
             iteration + 2 < low_iterations;
             ++iteration) {
          cp_async::wait_group<1>();
          __syncthreads();

          union LowScoreStorage {
            int32_t as_int[num_tiles_q][num_tiles_k][8];
            float as_float[num_tiles_q][num_tiles_k][8];
          } rs_storage;
          uint32_t q_offset = q_smem_mma;
          uint32_t k_offset = k_smem_mma;
          compute_int_qk<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              low_qk_inner, (HeadDim == 64 ? SwizzleMode::k64B
                                           : SwizzleMode::k128B),
              HeadDim / kPackInt8, DataType::kInt8>(
              smem_q8, smem_k8, rs_storage.as_int, q_offset, k_offset);
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; ++fk) {
#pragma unroll
              for (uint32_t element = 0; element < 8; ++element) {
                rs_storage.as_float[fq][fk][element] = __int2float_rz(
                    rs_storage.as_int[fq][fk][element]);
              }
            }
          }
          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(
              rs_storage.as_float, ro, m, d,
              base2_softmax_scale * current_dequant_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs_storage.as_float, d);
          uint32_t rs_fp8[num_tiles_q][num_tiles_k / 2][4];
          RS_32_to_8<num_tiles_q, num_tiles_k>(
              rs_storage.as_float, rs_fp8);

          // All stages reached here precede the final active stage, hence K
          // is a full physical tile and the donor unpredicated copy is safe.
          __syncthreads();
          absolute_stage += static_cast<uint32_t>(*low_delta++);
          k_lane =
              k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                    absolute_stage * kCtaK +
                    kCtaK / num_warps * warp_id +
                    lane_id / qk_line_lanes) *
                       HeadDim +
              (lane_id % qk_line_lanes) * kPackInt8;
          load_global_to_share<
              qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
              k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                               : SwizzleMode::k128B),
              HeadDim / kPackInt8, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_k8);
          cp_async::commit_group();

          cp_async::wait_group<1>();
          __syncthreads();
          compute_fp8_sv_inst_buf_fp16_accu<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              num_tiles_v, SwizzleMode::k64B, kCtaK / kPackFp8>(
              smem_v8, rs_fp8, ro, d);

          __syncthreads();
          v_lane =
              reinterpret_cast<int8_t*>(v8) +
              ((batch_id * num_kv_heads + kv_head) * HeadDim +
               HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                  padded_kv_len +
              absolute_stage * kCtaK +
              (lane_id % v_line_lanes) * kPackFp8;
          load_fp8_V_global_to_share<
              v_line_lanes, v_lines_per_warp, v_smem_iters_row,
              v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
              kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
          cp_async::commit_group();
          current_dequant_scale =
              q_dequant_scale * k_scale_base[absolute_stage];
        }

        // The second-last active position pre-dequantizes its score before
        // softmax, matching donor instruction rounding exactly.  Its next K
        // copy is the only prefetched K that may be a physical tail.
        if (low_iterations > 1) {
          cp_async::wait_group<1>();
          __syncthreads();

          union LowScoreStorage {
            int32_t as_int[num_tiles_q][num_tiles_k][8];
            float as_float[num_tiles_q][num_tiles_k][8];
          } rs_storage;
          uint32_t q_offset = q_smem_mma;
          uint32_t k_offset = k_smem_mma;
          compute_int_qk<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              low_qk_inner, (HeadDim == 64 ? SwizzleMode::k64B
                                           : SwizzleMode::k128B),
              HeadDim / kPackInt8, DataType::kInt8>(
              smem_q8, smem_k8, rs_storage.as_int, q_offset, k_offset);
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; ++fk) {
#pragma unroll
              for (uint32_t element = 0; element < 8; ++element) {
                const float score = __int2float_rz(
                    rs_storage.as_int[fq][fk][element]);
                rs_storage.as_float[fq][fk][element] =
                    score * current_dequant_scale;
              }
            }
          }

          __syncthreads();
          absolute_stage += static_cast<uint32_t>(*low_delta++);
          k_lane =
              k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                    absolute_stage * kCtaK +
                    kCtaK / num_warps * warp_id +
                    lane_id / qk_line_lanes) *
                       HeadDim +
              (lane_id % qk_line_lanes) * kPackInt8;
          k_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / qk_line_lanes;
          load_global_to_share<
              qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
              k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                               : SwizzleMode::k128B),
              HeadDim / kPackInt8, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_k8, k_load_row, kv_len);
          cp_async::commit_group();

          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(
              rs_storage.as_float, ro, m, d, base2_softmax_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs_storage.as_float, d);
          uint32_t rs_fp8[num_tiles_q][num_tiles_k / 2][4];
          RS_32_to_8<num_tiles_q, num_tiles_k>(
              rs_storage.as_float, rs_fp8);

          __syncthreads();
          cp_async::wait_group<1>();
          __syncthreads();
          compute_fp8_sv_inst_buf_fp16_accu<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              num_tiles_v, SwizzleMode::k64B, kCtaK / kPackFp8>(
              smem_v8, rs_fp8, ro, d);

          __syncthreads();
          v_lane =
              reinterpret_cast<int8_t*>(v8) +
              ((batch_id * num_kv_heads + kv_head) * HeadDim +
               HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                  padded_kv_len +
              absolute_stage * kCtaK +
              (lane_id % v_line_lanes) * kPackFp8;
          load_fp8_V_global_to_share<
              v_line_lanes, v_lines_per_warp, v_smem_iters_row,
              v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
              kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
          cp_async::commit_group();
          current_dequant_scale =
              q_dequant_scale * k_scale_base[absolute_stage];
        }

        // Last active position: drain K first, apply the only tail mask, then
        // drain V after producing the stage's online-softmax/P fragment.
        {
          cp_async::wait_group<1>();
          __syncthreads();

          union LowScoreStorage {
            int32_t as_int[num_tiles_q][num_tiles_k][8];
            float as_float[num_tiles_q][num_tiles_k][8];
          } rs_storage;
          uint32_t q_offset = q_smem_mma;
          uint32_t k_offset = k_smem_mma;
          compute_int_qk<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              low_qk_inner, (HeadDim == 64 ? SwizzleMode::k64B
                                           : SwizzleMode::k128B),
              HeadDim / kPackInt8, DataType::kInt8>(
              smem_q8, smem_k8, rs_storage.as_int, q_offset, k_offset);
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; ++fk) {
#pragma unroll
              for (uint32_t element = 0; element < 8; ++element) {
                const float score = __int2float_rz(
                    rs_storage.as_int[fq][fk][element]);
                rs_storage.as_float[fq][fk][element] =
                    score * current_dequant_scale;
              }
            }
          }
          const uint32_t k_index_base =
              absolute_stage * kCtaK +
              get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
              2 * (lane_id % 4);
          apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
              k_index_base, rs_storage.as_float, kv_len);
          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(
              rs_storage.as_float, ro, m, d, base2_softmax_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs_storage.as_float, d);
          uint32_t rs_fp8[num_tiles_q][num_tiles_k / 2][4];
          RS_32_to_8<num_tiles_q, num_tiles_k>(
              rs_storage.as_float, rs_fp8);

          cp_async::wait_group<0>();
          __syncthreads();
          compute_fp8_sv_inst_buf_fp16_accu<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              num_tiles_v, SwizzleMode::k64B, kCtaK / kPackFp8>(
              smem_v8, rs_fp8, ro, d);
          __syncthreads();
        }
      } else {
        // Mixed specializations retain their separate stage-by-stage path;
        // this change intentionally does not perturb their spill structure.
        for (uint32_t iteration = 0; iteration < low_iterations; ++iteration) {
#if defined(MPA_K64_BLOCK_MODE)
          // Native K64 routes store absolute physical-stage IDs.  The donor
          // Q128 path uses positive deltas so that its tightly pipelined
          // all-FP8 loop can update one running stage index instead.
          absolute_stage = static_cast<uint32_t>(low_lut[iteration]);
#else
          absolute_stage += static_cast<uint32_t>(low_lut[iteration]);
#endif

          int8_t* k_lane =
              k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                    absolute_stage * kCtaK +
                    kCtaK / num_warps * warp_id +
                    lane_id / qk_line_lanes) *
                       HeadDim +
              (lane_id % qk_line_lanes) * kPackInt8;
          uint32_t k_smem_load = smem_k8.get_permuted_offset(
              warp_id * qk_lines_per_warp * k_smem_iters_col +
                  lane_id / qk_line_lanes,
              lane_id % qk_line_lanes);
          const uint32_t k_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / qk_line_lanes;
          load_global_to_share<
              qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
              k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                               : SwizzleMode::k128B),
              HeadDim / kPackInt8, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_k8, k_load_row, kv_len);
          cp_async::commit_group();

          int8_t* v_lane =
              reinterpret_cast<int8_t*>(v8) +
              ((batch_id * num_kv_heads + kv_head) * HeadDim +
               HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                  padded_kv_len +
              absolute_stage * kCtaK +
              (lane_id % v_line_lanes) * kPackFp8;
          uint32_t v_smem_load = smem_v8.get_permuted_offset(
              warp_id * v_lines_per_warp * v_smem_iters_col +
                  lane_id / v_line_lanes,
              lane_id % v_line_lanes);
          load_fp8_V_global_to_share<
              v_line_lanes, v_lines_per_warp, v_smem_iters_row,
              v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
              kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
          cp_async::commit_group();
          cp_async::wait_group<0>();
          __syncthreads();

          union LowScoreStorage {
            int32_t as_int[num_tiles_q][num_tiles_k][8];
            float as_float[num_tiles_q][num_tiles_k][8];
          } rs_storage;
          uint32_t q_offset = q_smem_mma;
          uint32_t k_offset = k_smem_mma;
          compute_int_qk<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              low_qk_inner, (HeadDim == 64 ? SwizzleMode::k64B
                                           : SwizzleMode::k128B),
              HeadDim / kPackInt8, DataType::kInt8>(
              smem_q8, smem_k8, rs_storage.as_int, q_offset, k_offset);

          const float dequant_scale =
              q_dequant_scale *
              k_scale[(batch_id * num_kv_heads + kv_head) *
                          num_physical_stages +
                      absolute_stage];
          const bool donor_predequant_boundary =
              iteration + 2 >= low_iterations;
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; ++fk) {
#pragma unroll
              for (uint32_t element = 0; element < 8; ++element) {
                const float score =
                    __int2float_rz(rs_storage.as_int[fq][fk][element]);
                rs_storage.as_float[fq][fk][element] =
                    donor_predequant_boundary
                    ? score * dequant_scale
                    : score;
              }
            }
          }
          if (
#if defined(MPA_K64_BLOCK_MODE)
              true
#else
              iteration + 1 == low_iterations
#endif
          ) {
#if defined(MPA_K64_BLOCK_MODE)
            // A logical 2-D tile may occupy fewer than 64 lanes even when it
            // is not the final physical stage (8x7 is the common example).
            // Mask against this selected stage's exact valid count, not the
            // global padded K capacity.
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs_storage.as_float, valid_k);
#else
            const uint32_t k_index_base =
                absolute_stage * kCtaK +
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs_storage.as_float, kv_len);
#endif
          }
          const float stage_scale = donor_predequant_boundary
              ? base2_softmax_scale
              : base2_softmax_scale * dequant_scale;
          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(
              rs_storage.as_float, ro, m, d, stage_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs_storage.as_float, d);
          uint32_t rs_fp8[num_tiles_q][num_tiles_k / 2][4];
          RS_32_to_8<num_tiles_q, num_tiles_k>(
              rs_storage.as_float, rs_fp8);
          if constexpr (HeadDim == 128) {
            compute_fp8_sv_stage_tilewise<
                0, num_tiles_q, num_tiles_k, num_tiles_v,
                SwizzleMode::k64B, kCtaK / kPackFp8>(
                smem_v8, rs_fp8, ro);
          } else {
            compute_fp8_sv_inst_buf_fp16_accu<
                num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                num_tiles_v, SwizzleMode::k64B, kCtaK / kPackFp8>(
                smem_v8, rs_fp8, ro, d);
          }
          __syncthreads();
        }
      }
    }
  }

  if constexpr (HasFp16) {
    const bool need_q16 = high_iterations != 0;
    HalfQKVSmem<HeadDim> smem_q16(smem);

    constexpr uint32_t half_line_lanes = 8;
    constexpr uint32_t half_lines_per_warp = 4;
    constexpr uint32_t half_smem_iters_row =
        HeadDim / (half_line_lanes * kPackHalf);
    constexpr uint32_t q_half_smem_iters_col =
        kCtaQ / (num_warps * half_lines_per_warp);
    if (need_q16) {
      half* q_lane =
          q16 + ((batch_id * num_qo_heads + head_id) * qo_len +
                 query_block * kCtaQ +
                 kCtaQ / num_warps * warp_id + lane_id / half_line_lanes) *
                    HeadDim +
          (lane_id % half_line_lanes) * kPackHalf;
      uint32_t q_smem_load = smem_q16.get_permuted_offset(
          warp_id * half_lines_per_warp * q_half_smem_iters_col +
              lane_id / half_line_lanes,
          lane_id % half_line_lanes);
      const uint32_t q_load_row =
          query_block * kCtaQ + kCtaQ / num_warps * warp_id +
          lane_id / half_line_lanes;
      load_global_to_share<
          half_line_lanes, half_lines_per_warp, half_smem_iters_row,
          q_half_smem_iters_col, SwizzleMode::k128B, HeadDim / kPackHalf,
          kCtaQ>(
          q_lane, q_smem_load, HeadDim, smem_q16, q_load_row, qo_len);
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();
    }

    if constexpr (HasFp8 && SmoothK) {
      if (low_iterations != 0 && high_iterations != 0) {
          const half* mean_base =
              k_mean + (batch_id * num_kv_heads + kv_head) * HeadDim;
          restore_smoothed_low_rowmax<HeadDim, num_tiles_q>(
              smem_q16, mean_base, m, query_block, qo_len,
              base2_softmax_scale);
      }
    }
  }

  // Precision-boundary conversion: every executed FP8 phase leaves both the
  // probability-offset domain and the quantized-V domain before the final
  // normalize, including the compile-time FP8-only family.
  if constexpr (HasFp8) {
    if (low_iterations != 0) {
      const float* v_scale_base =
          v_scale + (batch_id * num_kv_heads + kv_head) * HeadDim;
      leave_fp8_probability_and_v_domains(ro, d, v_scale_base);
    }
  }

  if constexpr (HasFp16) {
    HalfQKVSmem<HeadDim> smem_q16(smem);
    HalfQKVSmem<HeadDim> smem_kv16(
        smem + kCtaQ * HeadDim * sizeof(half));

    constexpr uint32_t half_line_lanes = 8;
    constexpr uint32_t half_lines_per_warp = 4;
    constexpr uint32_t half_smem_iters_row =
        HeadDim / (half_line_lanes * kPackHalf);
    constexpr uint32_t kv_half_smem_iters_col =
        kCtaK / (num_warps * half_lines_per_warp);

    if (high_iterations != 0) {
      int32_t* high_lut = fp16_lut + metadata_row * num_physical_stages;
      const uint32_t q_smem_mma = smem_q16.get_permuted_offset(
          get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ + lane_id % 16,
          lane_id / 16);
      const uint32_t kv_smem_mma = smem_kv16.get_permuted_offset(
          lane_id % 8 + (lane_id / 16) * 8,
          (lane_id / 8) % 2);
      uint32_t absolute_stage = 0;

      for (uint32_t iteration = 0; iteration < high_iterations; ++iteration) {
#if defined(MPA_K64_BLOCK_MODE)
        // The H3 mixed route keeps one compact video list per row. FP8 reads
        // its leading segment. FP16 first executes the exact prefix stages,
        // then reads the following rescue segment after the row's variable
        // FP8 count. This avoids materializing a second full R x K LUT.
        absolute_stage = iteration < fp16_prefix_stages
            ? iteration
            : static_cast<uint32_t>(
                  high_lut[
                      low_iterations + iteration - fp16_prefix_stages]);
#else
        absolute_stage += static_cast<uint32_t>(high_lut[iteration]);
#endif

        if constexpr (HeadDim == 128 && num_tiles_q == 2) {
          if (iteration != 0) {
            reload_d128_fq0_q(
                smem_q16, q16, batch_id, head_id, num_qo_heads,
                query_block, qo_len);
          }
        }

        if constexpr (HeadDim == 128 && HasFp8 && num_tiles_q == 2) {
          const uint32_t k_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / half_line_lanes;
          const uint32_t kv_head_row =
              batch_id * num_kv_heads + kv_head;
          uint32_t k_logical_row;
          asm volatile(
              "mad.lo.u32 %0, %1, %2, %3;\n"
              : "=r"(k_logical_row)
              : "r"(kv_head_row), "r"(kv_len), "r"(k_load_row));
          half* k_lane =
              k16 + k_logical_row * HeadDim +
              (lane_id % half_line_lanes) * kPackHalf;
          uint32_t k_smem_load = smem_kv16.get_permuted_offset(
              warp_id * half_lines_per_warp * kv_half_smem_iters_col +
                  lane_id / half_line_lanes,
              lane_id % half_line_lanes);
          load_global_to_share<
              half_line_lanes, half_lines_per_warp, half_smem_iters_row,
              kv_half_smem_iters_col, SwizzleMode::k128B,
              HeadDim / kPackHalf, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_kv16, k_load_row, kv_len);
        } else {
          half* k_lane =
              k16 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                     absolute_stage * kCtaK +
                     kCtaK / num_warps * warp_id +
                     lane_id / half_line_lanes) *
                        HeadDim +
              (lane_id % half_line_lanes) * kPackHalf;
          uint32_t k_smem_load = smem_kv16.get_permuted_offset(
              warp_id * half_lines_per_warp * kv_half_smem_iters_col +
                  lane_id / half_line_lanes,
              lane_id % half_line_lanes);
          const uint32_t k_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / half_line_lanes;
          load_global_to_share<
              half_line_lanes, half_lines_per_warp, half_smem_iters_row,
              kv_half_smem_iters_col, SwizzleMode::k128B,
              HeadDim / kPackHalf, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_kv16, k_load_row, kv_len);
        }
        cp_async::commit_group();
        cp_async::wait_group<0>();
        __syncthreads();

        if constexpr (HeadDim == 128 && num_tiles_q == 2) {
          // fq=0 is stashed into its consumed Q rows; fq=1 remains in
          // registers.  This removes 32 live score registers at the QK peak
          // while keeping the CTA-lifetime FP32 m/d/O state in registers.
          float rs_active[1][num_tiles_k][8];
          uint32_t q_offset_fq0 = q_smem_mma;
          uint32_t k_offset_fq0 = kv_smem_mma;
          compute_half_qk<
              num_warps_q, num_warps_k, 1, num_tiles_k,
              half_qk_inner, SwizzleMode::k128B, HeadDim / kPackHalf>(
              smem_q16, smem_kv16, rs_active,
              q_offset_fq0, k_offset_fq0);
          if (
#if defined(MPA_K64_BLOCK_MODE)
              true
#else
              iteration + 1 == high_iterations
#endif
          ) {
#if defined(MPA_K64_BLOCK_MODE)
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<1, num_tiles_k>(
                k_index_base, rs_active, valid_k);
#else
            const uint32_t k_index_base =
                absolute_stage * kCtaK +
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<1, num_tiles_k>(
                k_index_base, rs_active, kv_len);
#endif
          }
          stash_d128_fq0_scores<num_tiles_k>(smem, rs_active);

          uint32_t q_offset_fq1 = smem_q16.get_permuted_offset(
              get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
                  kMmaQkM + lane_id % 16,
              lane_id / 16);
          uint32_t k_offset_fq1 = kv_smem_mma;
          compute_half_qk<
              num_warps_q, num_warps_k, 1, num_tiles_k,
              half_qk_inner, SwizzleMode::k128B, HeadDim / kPackHalf>(
              smem_q16, smem_kv16, rs_active,
              q_offset_fq1, k_offset_fq1);
          if (
#if defined(MPA_K64_BLOCK_MODE)
              true
#else
              iteration + 1 == high_iterations
#endif
          ) {
#if defined(MPA_K64_BLOCK_MODE)
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<1, num_tiles_k>(
                k_index_base, rs_active, valid_k);
#else
            const uint32_t k_index_base =
                absolute_stage * kCtaK +
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<1, num_tiles_k>(
                k_index_base, rs_active, kv_len);
#endif
          }

          // K and V share one 16-KiB scratch region.  V copy overlaps fq=1
          // softmax and P conversion; fq=0 is recovered after fq=1 PV.
          __syncthreads();
          const uint32_t v_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / half_line_lanes;
          const uint32_t kv_head_row =
              batch_id * num_kv_heads + kv_head;
          uint32_t v_logical_row;
          // Materialize V's full logical row at its use site.  Leaving the
          // equivalent K/V expression visible lets ptxas keep one row base
          // live across the entire D128 score/softmax/PV region and spill it.
          asm volatile(
              "mad.lo.u32 %0, %1, %2, %3;\n"
              : "=r"(v_logical_row)
              : "r"(kv_head_row), "r"(kv_len), "r"(v_load_row));
          half* v_lane =
              v16 + v_logical_row * HeadDim +
              (lane_id % half_line_lanes) * kPackHalf;
          uint32_t v_smem_load = smem_kv16.get_permuted_offset(
              warp_id * half_lines_per_warp * kv_half_smem_iters_col +
                  lane_id / half_line_lanes,
              lane_id % half_line_lanes);
          load_global_to_share<
              half_line_lanes, half_lines_per_warp, half_smem_iters_row,
              kv_half_smem_iters_col, SwizzleMode::k128B,
              HeadDim / kPackHalf, kCtaK>(
              v_lane, v_smem_load, HeadDim, smem_kv16, v_load_row, kv_len);
          cp_async::commit_group();

          update_mdo<1, num_tiles_k, num_tiles_v, false, false, false>(
              rs_active, ro + 1, m + 1, d + 1, base2_softmax_scale);
          accumulate_d<1, num_tiles_k, ComputeUnit::kCudaCore>(
              rs_active, d + 1);
          pack_rs_fp16_inplace<1, num_tiles_k>(rs_active);
          cp_async::wait_group<0>();
          __syncthreads();
          compute_fp16_sv_stage_tilewise<
              0, 1, num_tiles_k, num_tiles_v,
              SwizzleMode::k128B, HeadDim / kPackHalf, HasFp8>(
              smem_kv16, rs_active, ro + 1);

          // Reuse the same 32-register fragment for fq=0 instead of exposing
          // two arrays to ptxas in one scope.
          load_d128_fq0_scores<num_tiles_k>(smem, rs_active);
          update_mdo<1, num_tiles_k, num_tiles_v, false, false, false>(
              rs_active, ro, m, d, base2_softmax_scale);
          accumulate_d<1, num_tiles_k, ComputeUnit::kCudaCore>(rs_active, d);
          pack_rs_fp16_inplace<1, num_tiles_k>(rs_active);
          compute_fp16_sv_stage_tilewise<
              0, 1, num_tiles_k, num_tiles_v,
              SwizzleMode::k128B, HeadDim / kPackHalf, HasFp8>(
              smem_kv16, rs_active, ro);
        } else {
          float rs[num_tiles_q][num_tiles_k][8];
          uint32_t q_offset = q_smem_mma;
          uint32_t k_offset = kv_smem_mma;
          compute_half_qk<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              half_qk_inner, SwizzleMode::k128B, HeadDim / kPackHalf>(
              smem_q16, smem_kv16, rs, q_offset, k_offset);
          if (
#if defined(MPA_K64_BLOCK_MODE)
              true
#else
              iteration + 1 == high_iterations
#endif
          ) {
#if defined(MPA_K64_BLOCK_MODE)
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs, valid_k);
#else
            const uint32_t k_index_base =
                absolute_stage * kCtaK +
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs, kv_len);
#endif
          }

          __syncthreads();
          half* v_lane =
              v16 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                     absolute_stage * kCtaK +
                     kCtaK / num_warps * warp_id +
                     lane_id / half_line_lanes) *
                        HeadDim +
              (lane_id % half_line_lanes) * kPackHalf;
          uint32_t v_smem_load = smem_kv16.get_permuted_offset(
              warp_id * half_lines_per_warp * kv_half_smem_iters_col +
                  lane_id / half_line_lanes,
              lane_id % half_line_lanes);
          const uint32_t v_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / half_line_lanes;
          load_global_to_share<
              half_line_lanes, half_lines_per_warp, half_smem_iters_row,
              kv_half_smem_iters_col, SwizzleMode::k128B,
              HeadDim / kPackHalf, kCtaK>(
              v_lane, v_smem_load, HeadDim, smem_kv16, v_load_row, kv_len);
          cp_async::commit_group();

          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(
              rs, ro, m, d, base2_softmax_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs, d);
          pack_rs_fp16_inplace<num_tiles_q, num_tiles_k>(rs);
          cp_async::wait_group<0>();
          __syncthreads();
          compute_fp16_sv_stage_tilewise<
              0, num_tiles_q, num_tiles_k, num_tiles_v,
              SwizzleMode::k128B, HeadDim / kPackHalf>(
              smem_kv16, rs, ro);
        }
        __syncthreads();
      }
    }
  }

#if defined(MPA_K64_BLOCK_MODE)
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      float denominator = d[fq][row];
      denominator += __shfl_xor_sync(0xffffffff, denominator, 1);
      denominator += __shfl_xor_sync(0xffffffff, denominator, 2);
      m[fq][row] =
          (m[fq][row] + math::ptx_log2(denominator)) * math::log2e_recp;
      d[fq][row] = math::ptx_rcp(denominator);
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; ++fv) {
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        ro[fq][fv][element] *= d[fq][(element % 4) / 2];
      }
    }
  }
  if (lane_id % 4 == 0) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
      for (uint32_t row_half = 0; row_half < 2; ++row_half) {
        const uint32_t local_row =
            get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
            fq * kMmaQkM + lane_id / 4 + row_half * 8;
        const uint32_t global_row = query_block * kCtaQ + local_row;
        if (global_row < qo_len) {
          lse[(batch_id * num_qo_heads + head_id) * qo_len + global_row] =
              m[fq][row_half];
        }
      }
    }
  }
#else
  normalize_d_inplace<num_tiles_q, num_tiles_v>(ro, d);
#endif

  OutputSmem<HeadDim> smem_output(smem);
  const uint32_t output_row_base =
      get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; ++fv) {
      uint32_t output_offset = smem_output.get_permuted_offset(
          output_row_base + fq * kMmaQkM,
          fv * (kMmaPvN / kPackHalf));
      Half2Bits packed_half;
      packed_half.value =
          __float22half2_rn(reinterpret_cast<float2*>(ro[fq][fv])[0]);
      reinterpret_cast<uint32_t*>(smem_output.base + output_offset)
          [lane_id % 4] =
          packed_half.bits;
      packed_half.value =
          __float22half2_rn(reinterpret_cast<float2*>(ro[fq][fv])[1]);
      reinterpret_cast<uint32_t*>(
          smem_output.base + output_offset + 8 * (HeadDim / kPackHalf))
          [lane_id % 4] =
          packed_half.bits;
      output_offset = smem_output.get_permuted_offset(
          output_row_base + fq * kMmaQkM,
          fv * (kMmaPvN / kPackHalf) + 1);
      packed_half.value =
          __float22half2_rn(reinterpret_cast<float2*>(ro[fq][fv])[2]);
      reinterpret_cast<uint32_t*>(smem_output.base + output_offset)
          [lane_id % 4] =
          packed_half.bits;
      packed_half.value =
          __float22half2_rn(reinterpret_cast<float2*>(ro[fq][fv])[3]);
      reinterpret_cast<uint32_t*>(
          smem_output.base + output_offset + 8 * (HeadDim / kPackHalf))
          [lane_id % 4] =
          packed_half.bits;
    }
  }
  __syncwarp();
  if constexpr (HeadDim == 128 && HasFp16 && !HasFp8) {
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lane_id));
  }

  constexpr uint32_t output_line_lanes = 8;
  constexpr uint32_t output_lines_per_warp = 4;
  constexpr uint32_t output_smem_iters_row =
      HeadDim / (output_line_lanes * kPackHalf);
  constexpr uint32_t output_smem_iters_col =
      kCtaQ / (num_warps * output_lines_per_warp);
  half* output_lane =
      output + ((batch_id * num_qo_heads + head_id) * qo_len +
                query_block * kCtaQ +
                kWarpQ * get_warp_idx_q<num_warps_q, num_warps_k>() +
                lane_id / output_line_lanes) *
                   HeadDim +
      (lane_id % output_line_lanes) * kPackHalf;
  uint32_t output_offset = smem_output.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
          lane_id / output_line_lanes,
      lane_id % output_line_lanes);
  uint32_t output_global_row =
      query_block * kCtaQ +
      kCtaQ / num_warps * warp_id + lane_id / output_line_lanes;
#pragma unroll
  for (uint32_t row_iter = 0; row_iter < output_smem_iters_col; ++row_iter) {
#pragma unroll
    for (uint32_t col_iter = 0; col_iter < output_smem_iters_row; ++col_iter) {
      if (output_global_row < qo_len) {
        smem_output.store_128b(output_offset, output_lane);
      }
      output_lane += output_line_lanes * kPackHalf;
      output_offset =
          smem_output.advance_offset_by_column<output_line_lanes>(output_offset);
    }
    output_offset = smem_output.advance_offset_by_row<output_lines_per_warp>(
        output_offset - output_smem_iters_row * output_line_lanes);
    output_lane += output_lines_per_warp * HeadDim -
                   output_smem_iters_row * output_line_lanes * kPackHalf;
    output_global_row += output_lines_per_warp;
  }
}

}  // namespace mpa::attention

template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
void MPA_ATTENTION_LAUNCH_ENTRY(
    int8_t* q8,
    int8_t* k8,
    __nv_fp8_e4m3* v8,
    half* q16,
    half* k16,
    half* v16,
    half* k_mean,
    half* output,
    int32_t* fp8_lut,
    int32_t* fp8_count,
    int32_t* fp16_lut,
    int32_t* fp16_count,
    float* q_scale,
    float* k_scale,
    float* v_scale,
#if defined(MPA_K64_BLOCK_MODE)
    const int32_t* valid_k_counts,
    float* lse,
    uint32_t fp16_prefix_stages,
#endif
    uint32_t batch_size,
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t padded_kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    float softmax_scale,
    cudaStream_t stream) {
  constexpr uint32_t low_smem_bytes =
      HasFp8 ? (mpa::attention::kCtaQ + 2 * mpa::attention::kCtaK) *
                   HeadDim
             : 0;
  constexpr uint32_t high_smem_bytes =
      HasFp16 ? (mpa::attention::kCtaQ + mpa::attention::kCtaK) *
                    HeadDim * sizeof(half)
              : 0;
  constexpr uint32_t output_smem_bytes =
      mpa::attention::kCtaQ * HeadDim * sizeof(half);
  constexpr uint32_t smem_bytes =
      std::max(std::max(low_smem_bytes, high_smem_bytes), output_smem_bytes);

  auto kernel =
      mpa::attention::MPA_ATTENTION_KERNEL_ENTRY<
          HeadDim, HasFp8, HasFp16, SmoothK>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  const dim3 grid(
      div_ceil(qo_len, mpa::attention::kCtaQ), num_qo_heads, batch_size);
  const dim3 block(
      32,
      (mpa::attention::kCtaQ / mpa::attention::kWarpQ) *
          (mpa::attention::kCtaK / mpa::attention::kWarpK));
  kernel<<<grid, block, smem_bytes, stream>>>(
      q8, k8, v8, q16, k16, v16, k_mean, output, fp8_lut, fp8_count,
      fp16_lut, fp16_count, q_scale, k_scale, v_scale,
#if defined(MPA_K64_BLOCK_MODE)
      valid_k_counts, lse, fp16_prefix_stages,
#endif
      qo_len, kv_len,
      padded_kv_len, num_qo_heads / num_kv_heads, softmax_scale);
}

#undef MPA_ATTENTION_KERNEL_ENTRY
#undef MPA_ATTENTION_LAUNCH_ENTRY
