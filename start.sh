#!/usr/bin/env bash
set -euo pipefail

# RunPod-compatible launcher:
# - starts sshd so RunPod SSH/web terminal remain usable
# - starts Pixal3D UI on 0.0.0.0:7860

mkdir -p /var/run/sshd /root/.ssh /workspace
chmod 700 /root/.ssh || true

# RunPod normally injects SSH keys itself for SSH-enabled pods. Do not create a password.
/usr/sbin/sshd || true

cd /app/Pixal3D

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn_3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export GRADIO_SHARE="${GRADIO_SHARE:-0}"

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/huggingface/hub}"
export TORCH_HOME="${TORCH_HOME:-/workspace/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/cache}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME"

exec python3.10 app.py
