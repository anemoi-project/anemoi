#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${EVG_SERVER_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
HUNYUAN_ROOT="${EVG_HUNYUAN_ROOT:-${PROJECT_ROOT}/third_party/HunyuanVideo-1.5}"
MODEL_PATH="${EVG_MODEL_PATH:?Set EVG_MODEL_PATH to the HunyuanVideo-1.5 checkpoint}"
RUN_TIMESTAMP="${EVG_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PROMPT="${EVG_PROMPT:-In the style of Dunhuang sculptures, A graceful deity, playing a pipa, dances lightly in a museum, with flowing garments.}"
GPU_ID="${EVG_GPU_ID:-0}"
PYTHON_BIN="${EVG_PYTHON:-python}"
DRAFT_ATTN="${EVG_DRAFT_ATTN:-true}"
VIDEO_LENGTH="${EVG_VIDEO_LENGTH:-33}"
NUM_INFERENCE_STEPS="${EVG_NUM_INFERENCE_STEPS:-8}"
SEED="${EVG_SEED:-1234}"
RUN_MODE="dense"
if [[ "${DRAFT_ATTN}" == "true" ]]; then
  RUN_MODE="draft_triton"
fi
OUTPUT_DIR="${EVG_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
OUTPUT_PATH="${EVG_OUTPUT_PATH:-${OUTPUT_DIR}/hunyuan15_${RUN_MODE}_${RUN_TIMESTAMP}_seed${SEED}_720p_dunhuang_${VIDEO_LENGTH}f_${NUM_INFERENCE_STEPS}steps.mp4}"
OFFLOADING="${EVG_OFFLOADING:-true}"
GROUP_OFFLOADING="${EVG_GROUP_OFFLOADING:-false}"
DRAFT_PROFILE="${EVG_DRAFT_PROFILE:-true}"
DRAFT_SCHEDULE_CONFIG="${EVG_DRAFT_SCHEDULE_CONFIG:-${REPO_ROOT}/configs/hunyuanvideo-1.5/draft_25dense_80sparse.json}"
DRAFT_SCHEDULE_ARGS=()
if [[ "${DRAFT_ATTN}" == "true" && -n "${DRAFT_SCHEDULE_CONFIG}" ]]; then
  DRAFT_SCHEDULE_ARGS=(--draft_schedule_config "${DRAFT_SCHEDULE_CONFIG}")
fi

cd "${HUNYUAN_ROOT}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

exec env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" generate.py \
  --model_path "${MODEL_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --prompt "${PROMPT}" \
  --resolution 720p \
  --aspect_ratio 16:9 \
  --video_length "${VIDEO_LENGTH}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --seed "${SEED}" \
  --rewrite false \
  --cfg_distilled false \
  --enable_step_distill false \
  --sparse_attn false \
  --draft_attn "${DRAFT_ATTN}" \
  "${DRAFT_SCHEDULE_ARGS[@]}" \
  --draft_dense_fraction 0.25 \
  --draft_sparsity_ratio 0.8 \
  --draft_pool_h 8 \
  --draft_pool_w 16 \
  --draft_q_chunk_size 64 \
  --draft_k_chunk_size 64 \
  --draft_backend triton \
  --draft_profile "${DRAFT_PROFILE}" \
  --use_sageattn false \
  --enable_cache false \
  --enable_torch_compile false \
  --sr false \
  --save_pre_sr_video false \
  --offloading "${OFFLOADING}" \
  --group_offloading "${GROUP_OFFLOADING}" \
  --overlap_group_offloading false \
  --dtype bf16
