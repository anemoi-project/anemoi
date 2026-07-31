#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${EVG_FULL_SHAPE:-0}" == "1" ]]; then
  exec "${PYTHON:-python3}" -m evg.cli.main draft-attn-smoke \
    --preset "${EVG_PRESET:-wan2.2-720p}" \
    --device "${EVG_DEVICE:-auto}" \
    --dtype "${EVG_DTYPE:-float32}" \
    --backend "${EVG_DRAFT_BACKEND:-auto}" \
    --full-shape \
    "$@"
fi

exec "${PYTHON:-python3}" -m evg.cli.main draft-attn-smoke \
  --preset "${EVG_PRESET:-wan2.2-720p}" \
  --device "${EVG_DEVICE:-auto}" \
  --dtype "${EVG_DTYPE:-float32}" \
  --backend "${EVG_DRAFT_BACKEND:-auto}" \
  "$@"
