FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    ATTN_BACKEND=flash_attn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    TORCH_HOME=/runpod-volume/torch \
    XDG_CACHE_HOME=/runpod-volume/cache \
    MAX_JOBS=4 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    ca-certificates \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
 && rm -rf /var/lib/apt/lists/*

RUN git lfs install || true

RUN python3.10 -m pip install --upgrade pip setuptools wheel packaging ninja

# TRELLIS.2 currently documents PyTorch 2.6.0 + CUDA 12.4 as its default setup.
RUN python3.10 -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /app/TRELLIS.2

WORKDIR /app/TRELLIS.2

# TRELLIS.2's setup.sh exits if nvidia-smi is not present. Cloud Docker builders
# usually have the CUDA toolkit but no attached GPU, so install the same deps directly.
RUN python3.10 -m pip install \
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    opencv-python-headless \
    ninja \
    trimesh \
    transformers \
    gradio==6.0.1 \
    tensorboard \
    pandas \
    lpips \
    zstandard \
    kornia \
    timm \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# setup.sh installs pillow-simd after libjpeg-dev. Keep this non-fatal because
# some Python/package combinations may already have a compatible Pillow wheel.
RUN python3.10 -m pip uninstall -y pillow || true \
 && python3.10 -m pip install pillow-simd || python3.10 -m pip install pillow

RUN mkdir -p /tmp/extensions \
 && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
 && python3.10 -m pip install /tmp/extensions/nvdiffrast --no-build-isolation

RUN mkdir -p /tmp/extensions \
 && git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec \
 && python3.10 -m pip install /tmp/extensions/nvdiffrec --no-build-isolation

RUN mkdir -p /tmp/extensions \
 && git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive \
 && python3.10 -m pip install /tmp/extensions/CuMesh --no-build-isolation

RUN mkdir -p /tmp/extensions \
 && git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive \
 && python3.10 -m pip install /tmp/extensions/FlexGEMM --no-build-isolation

RUN mkdir -p /tmp/extensions \
 && cp -r /app/TRELLIS.2/o-voxel /tmp/extensions/o-voxel \
 && python3.10 -m pip install /tmp/extensions/o-voxel --no-build-isolation

RUN python3.10 -m pip install flash-attn==2.7.3 --no-build-isolation

# Pixal3D currently references this Python 3.10 / Torch 2.6 / CUDA 12.4 NATTEN wheel.
RUN python3.10 -m pip install \
    "https://github.com/LDYang694/Storages/releases/download/20260430/natten-0.21.0+torch2.6cu124-cp310-cp310-linux_x86_64.whl"

WORKDIR /app

RUN git clone -b master https://github.com/TencentARC/Pixal3D.git /app/Pixal3D

WORKDIR /app/Pixal3D

RUN python3.10 -m pip install \
    "git+https://github.com/microsoft/MoGe.git" \
    diffusers==0.37.1 \
    accelerate==1.13.0 \
    gradio \
    plyfile==1.1.3 \
    spaces \
    requests \
    runpod

RUN python3.10 -m pip install --force-reinstall --no-deps \
    "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"

# Pixal3D hardcodes flash_attn_3 in its scripts at the time this image was authored.
# Default to FlashAttention 2, while still allowing ATTN_BACKEND=flash_attn_3 on H100/Hopper builds.
RUN python3.10 - <<'PY'
from pathlib import Path

for name in ["inference.py", "app.py"]:
    p = Path(name)
    if not p.exists():
        continue

    text = p.read_text()
    text = text.replace(
        'os.environ["ATTN_BACKEND"] = "flash_attn_3"',
        'os.environ["ATTN_BACKEND"] = os.environ.get("ATTN_BACKEND", "flash_attn")'
    )
    p.write_text(text)
PY

COPY handler.py /app/handler.py

WORKDIR /app

CMD ["python3.10", "-u", "handler.py"]
