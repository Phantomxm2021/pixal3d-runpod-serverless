FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    ATTN_BACKEND=flash_attn_3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    TORCH_HOME=/runpod-volume/torch \
    XDG_CACHE_HOME=/runpod-volume/cache \
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
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
 && rm -rf /var/lib/apt/lists/*

RUN git lfs install || true
RUN python3.10 -m pip install --upgrade pip setuptools wheel packaging

# Pixal3D README: follow TRELLIS.2 first, then install Pixal3D deps.
# For the web demo/H-series GPU image, Pixal3D provides requirements-hfdemo.txt
# with prebuilt wheels for flash_attn_3, cumesh, flex_gemm, o_voxel,
# nvdiffrast, and nvdiffrec_render.
RUN git clone -b master https://github.com/TencentARC/Pixal3D.git /app/Pixal3D

WORKDIR /app/Pixal3D

RUN python3.10 -m pip install -r requirements-hfdemo.txt

# README Step 3: install Pixal3D's required utils3d wheel.
RUN python3.10 -m pip install --force-reinstall --no-deps \
    "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"

# RunPod UI compatibility:
# - the upstream app hardcodes share=True and default Gradio binding
# - bind to 0.0.0.0:7860 for RunPod HTTP proxy
# - keep the upstream flash_attn_3 default, matching requirements-hfdemo.txt
RUN python3.10 - <<'PY'
from pathlib import Path

p = Path('app.py')
text = p.read_text()
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

# Fail the build immediately if Pixal3D's Gradio Server dependency is not present.
RUN python3.10 - <<'PY'
import gradio
from gradio import Server
print('Gradio version:', gradio.__version__)
print('Gradio Server import OK:', Server)
PY

COPY handler.py /app/handler.py
COPY run_ui.sh /app/run_ui.sh
RUN chmod +x /app/run_ui.sh

WORKDIR /app

CMD ["python3.10", "-u", "handler.py"]
