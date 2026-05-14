# Pixal3D RunPod Serverless worker.
# Built on top of Pixal3D's `requirements-hfdemo.txt`, which ships prebuilt
# cp310 wheels for every native CUDA dep (natten, flash_attn_3, cumesh,
# flex_gemm, o_voxel, nvdiffrast, nvdiffrec_render) — no source compilation.
# FA2 added separately for non-Hopper GPUs; handler picks FA2/FA3 at startup.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    TORCH_HOME=/runpod-volume/torch \
    XDG_CACHE_HOME=/runpod-volume/cache \
    TRITON_CACHE_DIR=/runpod-volume/triton \
    FLEX_GEMM_AUTOTUNE_CACHE_PATH=/runpod-volume/flex_gemm_autotune.json \
    PIXAL3D_MODEL_PATH=TencentARC/Pixal3D \
    MOGE_MODEL_PATH=Ruicheng/moge-2-vitl \
    BIREFNET_MODEL_PATH=ZhengPeng7/BiRefNet \
    PIXAL3D_BAKED_HF_CACHE=/opt/hf_baked/hub \
    PIXAL3D_TIMEOUT=900 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        git \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

# Pin Pixal3D to a specific ref for reproducibility. Override at build time:
#   docker build --build-arg PIXAL3D_REF=<commit-sha> .
ARG PIXAL3D_REF=master
RUN git clone -b ${PIXAL3D_REF} --depth 1 \
        https://github.com/TencentARC/Pixal3D.git /app/Pixal3D

WORKDIR /app/Pixal3D

# Gold-path install: prebuilt cp310 wheels for every native dep, plus torch
# 2.6.0+cu124 via PyTorch's index. Ships FA3 wheel (Hopper-only).
RUN python3.10 -m pip install -r requirements-hfdemo.txt

# FlashAttention 2 for Ampere/Ada (A100/A40/4090/L40S). Coexists with FA3 —
# different package name. Handler chooses backend by GPU capability at boot.
RUN python3.10 -m pip install \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Pixal3D's required utils3d override (per upstream README).
RUN python3.10 -m pip install --force-reinstall --no-deps \
        https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# RunPod SDK + huggingface_hub. transformers/diffusers/etc are in hfdemo.
RUN python3.10 -m pip install runpod huggingface_hub

# Bake BiRefNet (ungated, ~880 MB) into the image. The handler seeds the
# network volume cache from this path at first boot — removes BiRefNet from
# the HF download path entirely.
RUN python3.10 -c "import os; from huggingface_hub import snapshot_download; \
        snapshot_download(os.environ['BIREFNET_MODEL_PATH'], cache_dir=os.environ['PIXAL3D_BAKED_HF_CACHE'])"

WORKDIR /app

COPY handler.py /app/handler.py
COPY scripts /app/scripts

# Smoke-test that the install resolves. Catches wheel/ABI mismatches at
# build time instead of at the first RunPod worker cold start.
RUN python3.10 -c "import torch, runpod, flash_attn, transformers, diffusers, accelerate, huggingface_hub; \
        print('torch', torch.__version__); \
        print('cuda available (build-time, expected False):', torch.cuda.is_available())"

CMD ["python3.10", "/app/handler.py"]
