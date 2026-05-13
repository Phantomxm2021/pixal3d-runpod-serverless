FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    ATTN_BACKEND=flash_attn_3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/huggingface/hub \
    TORCH_HOME=/workspace/torch \
    XDG_CACHE_HOME=/workspace/cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SHARE=0 \
    PIP_NO_CACHE_DIR=1

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
RUN python3.10 -m pip install --upgrade pip setuptools wheel packaging

RUN git clone -b master https://github.com/TencentARC/Pixal3D.git /app/Pixal3D

WORKDIR /app/Pixal3D

# Pixal3D's Hugging Face demo / H-series install path. This uses prebuilt CUDA 12.4
# wheels for flash_attn_3, natten, cumesh, flex_gemm, o_voxel, nvdiffrast, and
# nvdiffrec_render, avoiding source builds in Docker.
RUN python3.10 -m pip install -r requirements-hfdemo.txt

# app.py imports `spaces`, but the normal requirements file does not include it.
# Keep huggingface_hub below 1.0 because transformers 4.57.x requires <1.0.
RUN python3.10 -m pip install \
    spaces \
    "huggingface_hub>=0.34.0,<1.0"

RUN python3.10 -m pip install --force-reinstall --no-deps \
    https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# RunPod UI compatibility only:
# - app.py already defaults to flash_attn_3; make it respect ATTN_BACKEND env var
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
        'os.environ["ATTN_BACKEND"] = os.environ.get("ATTN_BACKEND", "flash_attn_3")'
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
