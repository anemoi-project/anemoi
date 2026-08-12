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
 * Project-owned 8x16 raster-pooling boundary. No donor implementation
 * source is copied into this component.
 */

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
pool_qk_and_kmean(
    torch::Tensor q,
    torch::Tensor k,
    int64_t frames,
    int64_t height,
    int64_t width,
    bool compute_k_mean,
    int64_t group_patches);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
pool_qkv_and_kmean_vsum(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t frames,
    int64_t height,
    int64_t width,
    bool compute_k_mean,
    int64_t group_patches);

torch::Tensor draft_probability(
    torch::Tensor q_pool,
    torch::Tensor k_pool);

torch::Tensor draft_scores(
    torch::Tensor q_pool,
    torch::Tensor k_pool);

// Materialize the scaled pooled Q*K^T logits without the Draft softmax.  This
// is the input to the Sol-compatible per-row statistical router.
torch::Tensor draft_logits(
    torch::Tensor q_pool,
    torch::Tensor k_pool);

torch::Tensor draft_skip_error_risk(
    torch::Tensor q_pool,
    torch::Tensor k_pool,
    torch::Tensor qk_patch_statistics,
    torch::Tensor v_sum,
    int64_t frames,
    int64_t height,
    int64_t width);

std::tuple<torch::Tensor, torch::Tensor> draft_probability_with_lse(
    torch::Tensor q_pool,
    torch::Tensor k_pool);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
skip_tail_compensation(
    torch::Tensor probability_fp16,
    torch::Tensor row_lse_fp32,
    torch::Tensor precision_map,
    torch::Tensor v_sum_fp32,
    int64_t frames,
    int64_t height,
    int64_t width);

std::tuple<torch::Tensor, torch::Tensor> apply_token_skip_compensation(
    torch::Tensor q,
    torch::Tensor output,
    torch::Tensor lse_fp32,
    torch::Tensor k_pool_fp16,
    torch::Tensor v_sum_fp32,
    torch::Tensor precision_map,
    int64_t frames,
    int64_t height,
    int64_t width);

std::tuple<torch::Tensor, torch::Tensor> apply_k64_token_skip_compensation(
    torch::Tensor q_fp16,
    torch::Tensor output_fp16,
    torch::Tensor lse_fp32,
    torch::Tensor k_pool_fp16,
    torch::Tensor v_sum_fp32,
    torch::Tensor selected_map,
    torch::Tensor valid_k_counts,
    double softmax_scale);

// Optimized counterparts kept as separate dispatcher entries so the legacy
// one-token/CTA implementation remains available for controlled A/B checks.
std::tuple<torch::Tensor, torch::Tensor>
apply_k64_token_skip_compensation_warp(
    torch::Tensor q_fp16,
    torch::Tensor output_fp16,
    torch::Tensor lse_fp32,
    torch::Tensor k_pool_fp16,
    torch::Tensor v_sum_fp32,
    torch::Tensor selected_map,
    torch::Tensor valid_k_counts,
    double softmax_scale);

std::tuple<torch::Tensor, torch::Tensor>
apply_k64_qmean_skip_compensation(
    torch::Tensor q_pool_fp16,
    torch::Tensor output_fp16,
    torch::Tensor lse_fp32,
    torch::Tensor k_pool_fp16,
    torch::Tensor v_sum_fp32,
    torch::Tensor selected_map,
    torch::Tensor valid_k_counts,
    double softmax_scale);

torch::Tensor build_skip_centroid_kv(
    torch::Tensor k_pool_fp16,
    torch::Tensor v_sum_fp32,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t raster_pool_width);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_k64_route_metadata(
    torch::Tensor packed_k_fp16,
    torch::Tensor packed_v_fp16,
    torch::Tensor k_pool128_fp16,
    torch::Tensor v_sum128_fp32,
    int64_t frames,
    int64_t height,
    int64_t width);

torch::Tensor draft_skip_error_risk_k64(
    torch::Tensor q_pool,
    torch::Tensor k_pool64,
    torch::Tensor qk_patch_statistics,
    torch::Tensor k_variance64,
    torch::Tensor v_sum64,
    int64_t frames,
    int64_t height,
    int64_t width);

torch::Tensor blend_k64_route_scores(
    torch::Tensor k128_score_fp16,
    torch::Tensor k64_score_fp16,
    double alpha);

torch::Tensor draft_spatial_skip_error_risk_q4_k8(
    torch::Tensor packed_q_fp16,
    torch::Tensor packed_k_fp16,
    torch::Tensor packed_v_fp16,
    torch::Tensor base_risk_fp16,
    int64_t frames,
    int64_t height,
    int64_t width,
    double residual_coefficient);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision(
    torch::Tensor probability_fp16,
    int64_t n16,
    int64_t n8,
    int64_t n4);

// Exact per-head global probability routing with mandatory legal same-frame
// 2-D cross anchors.  Anchors consume seats from the unchanged global budget;
// all remaining seats retain the ordinary probability ranking.
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision_spatial_cross(
    torch::Tensor probability_fp16,
    int64_t n16,
    int64_t n8,
    int64_t n4,
    int64_t frames,
    int64_t patches_h,
    int64_t patches_w);

// Integration-private FP8/FP16 phase redistribution.  Ranking and the total
// retained count remain per head; only the FP16 prefix length may differ
// across [B,H].  The tensor counts are trusted stream-ordered metadata built
// by the project backend, not part of the stable public operator ABI.
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
    int64_t keep);

// Sol-compatible data-dependent per-row routing over pooled logits.  Threshold
// type 0 is diagonal K-centroid covariance and type 1 is the exact projected
// covariance (equivalently the empirical mean/std of each pooled-logit row).
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
    double low8_tau,
    double fp16_tau,
    int64_t threshold_type,
    int64_t local_fp16_radius,
    int64_t forced_sink_blocks,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t raster_pool_width,
    bool partial_logmass);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision_k64(
    torch::Tensor score_fp16,
    int64_t n16,
    int64_t n8,
    int64_t n4);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_sol_scores(
    torch::Tensor scores_fp16,
    double beta,
    double retained_fp16_ratio,
    double retained_fp8_ratio,
    double retained_fp4_ratio,
    bool force_dense);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
route_precision_segmented_control(
    torch::Tensor probability_fp16,
    int64_t n16,
    int64_t n8,
    int64_t n4);

torch::Tensor validate_route_metadata(
    torch::Tensor fp8_counts,
    torch::Tensor fp16_counts,
    torch::Tensor fp4_counts,
    torch::Tensor first_empty_row,
    int64_t n8,
    int64_t n16,
    int64_t n4);
