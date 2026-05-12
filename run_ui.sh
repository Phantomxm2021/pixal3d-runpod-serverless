#!/usr/bin/env bash
set -euo pipefail

cd /app/Pixal3D

export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export GRADIO_SHARE="${GRADIO_SHARE:-0}"

python3.10 app.py
