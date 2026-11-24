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
        gcc \
        python3.10-dev \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

# Pin Pixal3D to a specific ref for reproducibility. Override at build time:
#   docker build --build-arg PIXAL3D_REF=<commit-sha> .
ARG PIXAL3D_REF=f7cf38429b0bd264f1995f0f8743a88b1c728b94
RUN git init /app/Pixal3D \
    && git -C /app/Pixal3D fetch --depth 1 https://github.com/TencentARC/Pixal3D.git "$PIXAL3D_REF" \
    && git -C /app/Pixal3D checkout --detach FETCH_HEAD \
    && rm -rf /app/Pixal3D/.git

WORKDIR /app/Pixal3D

COPY requirements.txt /app/requirements.txt

# MoGe 3 introduces a source-built FlexGEMM dependency and requires a newer
# trimesh. The worker uses the MoGe 2 camera estimator.
ARG MOGE_REF=9642cf99a13fd829a4f7934fd8dab17e7091da9d
RUN sed -E '/^git\+https:\/\/github\.com\/microsoft\/MoGe\.git/d; /^gradio$/d' \
        requirements-hfdemo.txt > /tmp/worker-requirements.txt \
    && python3.10 -m pip install -r /tmp/worker-requirements.txt -r /app/requirements.txt \
        'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' \
        "moge @ git+https://github.com/microsoft/MoGe.git@${MOGE_REF}" \
    && rm /tmp/worker-requirements.txt

# FlashAttention 2 for Ampere/Ada (A100/A40/4090/L40S). Coexists with FA3 —
# different package name. Handler chooses backend by GPU capability at boot.
RUN python3.10 -m pip install \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Pixal3D's required utils3d override (per upstream README).
RUN python3.10 -m pip install --force-reinstall --no-deps \
        https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# Load the public background-removal model directly from the image at runtime.
RUN python3.10 -c "from huggingface_hub import snapshot_download; \
        snapshot_download('ZhengPeng7/BiRefNet', local_dir='/opt/birefnet', allow_patterns=['*.json', '*.py', '*.safetensors'])" \
    && rm -rf /opt/birefnet/.cache

WORKDIR /app

COPY rp_handler.py handler.py inference_worker.py /app/
COPY scripts /app/scripts
COPY patches/pixal3d-shared-dinov3.patch /tmp/pixal3d-shared-dinov3.patch

# Runpod's image builder has no GPU. Importing o_voxel or the Pixal3D
# pipeline initializes FlexGEMM's Triton driver during import. Check that
# those modules are present here and load them on the GPU worker at startup.
RUN git -C /app/Pixal3D apply /tmp/pixal3d-shared-dinov3.patch \
    && rm /tmp/pixal3d-shared-dinov3.patch \
    && python3.10 -m pip check \
    && PYTHONPATH=/app/Pixal3D python3.10 -c "from importlib.util import find_spec; \
        import torch, runpod, boto3, flash_attn, flash_attn_interface, natten; \
        from moge.model.v2 import MoGeModel; \
        from transformers import DINOv3ViTModel; \
        missing = [name for name in ('o_voxel', 'flex_gemm', 'pixal3d') if find_spec(name) is None]; \
        assert not missing, f'Missing worker modules: {missing}'; \
        assert torch.version.cuda == '12.4', f'Unexpected PyTorch CUDA version: {torch.version.cuda}'" \
    && python3.10 -c "from handler import handler; assert 'error' in handler({'input': {}})"

CMD ["python3", "-u", "-m", "scripts.worker_diagnostics"]
