FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    ATTN_BACKEND=flash_attn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/huggingface/hub \
    TORCH_HOME=/workspace/torch \
    XDG_CACHE_HOME=/workspace/cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SHARE=0 \
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
    openssh-server \
    bash \
    procps \
    nano \
    vim \
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

RUN mkdir -p /var/run/sshd /root/.ssh \
 && chmod 700 /root/.ssh \
 && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
 && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
 && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && echo 'ClientAliveInterval 60' >> /etc/ssh/sshd_config \
 && echo 'ClientAliveCountMax 3' >> /etc/ssh/sshd_config

RUN git lfs install || true
RUN python3.10 -m pip install --upgrade pip setuptools wheel packaging ninja

RUN python3.10 -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /app/TRELLIS.2

WORKDIR /app/TRELLIS.2

RUN python3.10 -m pip install \
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    opencv-python-headless \
    ninja \
    trimesh \
    transformers \
    gradio \
    tensorboard \
    pandas \
    lpips \
    zstandard \
    kornia \
    timm \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

RUN python3.10 -m pip install pillow-simd || python3.10 -m pip install pillow

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

WORKDIR /app
RUN git clone -b master https://github.com/TencentARC/Pixal3D.git /app/Pixal3D

WORKDIR /app/Pixal3D

RUN python3.10 -m pip install --no-build-isolation -r requirements.txt
RUN python3.10 -m pip install spaces
RUN python3.10 -m pip install --force-reinstall --no-deps \
    https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# RunPod UI compatibility only:
# - normal TRELLIS.2 setup uses flash_attn, so do not let app.py force flash_attn_3
# - bind Gradio to 0.0.0.0:7860 for RunPod HTTP proxy
# - disable Gradio share unless GRADIO_SHARE=1
RUN python3.10 - <<'PY'
from pathlib import Path

for filename in ["app.py", "inference.py"]:
    p = Path(filename)
    if not p.exists():
        continue
    text = p.read_text()
    text = text.replace(
        'os.environ["ATTN_BACKEND"] = "flash_attn_3"',
        'os.environ["ATTN_BACKEND"] = os.environ.get("ATTN_BACKEND", "flash_attn")'
    )
    if filename == "app.py":
        text = text.replace(
            "app.launch(show_error=True, share=True)",
            "app.launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '0.0.0.0'), server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')), show_error=True, share=os.environ.get('GRADIO_SHARE', '0') == '1')"
        )
        text = text.replace(
            "app.launch(show_error=True)",
            "app.launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '0.0.0.0'), server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')), show_error=True, share=os.environ.get('GRADIO_SHARE', '0') == '1')"
        )
    p.write_text(text)
PY

COPY run_ui.sh /app/run_ui.sh
COPY start.sh /app/start.sh
RUN chmod +x /app/run_ui.sh /app/start.sh

EXPOSE 22 7860
CMD ["/app/start.sh"]
