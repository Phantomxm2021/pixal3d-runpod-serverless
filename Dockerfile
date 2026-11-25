# Pixal3D RunPod Serverless worker.
# Use Pixal3D's prebuilt CUDA dependencies and build an Ampere/Ada NATTEN wheel.
# FA2 added separately for non-Hopper GPUs; handler picks FA2/FA3 at startup.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS natten-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates python3.10 python3-pip python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir 'cmake==3.31.6' setuptools wheel \
    && python3.10 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124'

# Compile only SM80 to fit the Runpod image build time limit. SM80 also runs on Ada.
# Pin the source archive and inspect the actual cubins without a GPU.
ARG NATTEN_N_WORKERS=4
RUN NATTEN_CUDA_ARCH='8.0' NATTEN_N_WORKERS="${NATTEN_N_WORKERS}" NATTEN_IS_BUILDING_DIST=1 \
    python3.10 -m pip wheel --verbose --no-cache-dir --no-deps --no-build-isolation --wheel-dir /wheels \
        'https://files.pythonhosted.org/packages/46/c9/aa6bd171aa23de07f401131ce2aff0d1cfa036e22ad8e7e8faa9d66b8067/natten-0.21.0.tar.gz#sha256=810aaf179e1a6fdb0a9279d7858d00412a0d3f045e546cada415205215ccd539' \
    && python3.10 -m zipfile -e /wheels/natten-*.whl /tmp/natten-check \
    && cuobjdump --list-elf /tmp/natten-check/natten/libnatten*.so > /tmp/natten-cubins.txt \
    && grep -oE 'sm_[0-9]+a?' /tmp/natten-cubins.txt | sort -u > /tmp/natten-archs.txt \
    && cat /tmp/natten-archs.txt \
    && grep -qx sm_80 /tmp/natten-archs.txt \
    && rm -rf /tmp/natten-check /tmp/natten-cubins.txt /tmp/natten-archs.txt

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
RUN --mount=type=bind,from=natten-build,source=/wheels,target=/tmp/natten-wheels \
    sed -E '/^git\+https:\/\/github\.com\/microsoft\/MoGe\.git/d; /^gradio$/d; /^natten([=<>!~ @]|$)/d; /\/natten-[^/]*\.whl([?#].*)?$/d' \
        requirements-hfdemo.txt > /tmp/worker-requirements.txt \
    && python3.10 -m pip install -r /tmp/worker-requirements.txt -r /app/requirements.txt \
        'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' \
        /tmp/natten-wheels/natten-*.whl \
        "moge @ git+https://github.com/microsoft/MoGe.git@${MOGE_REF}" \
    && rm /tmp/worker-requirements.txt

# Pixal3D's upstream NATTEN wheel contains SM90 cubins. Keep it isolated from
# the system SM80 package and select it on Hopper before importing the pipeline.
RUN python3.10 -m pip install --no-deps --target /opt/natten-hopper \
    'https://github.com/LDYang694/Storages/releases/download/20260430/natten-0.21.0+torch2.6cu124-cp310-cp310-linux_x86_64.whl#sha256=6efb93997931d8efdb4fa45a4ccd5f11108f6cd8adc876669b6892839b388247'

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

RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bundle the public camera checkpoint to avoid Global Volume reads during inference.
RUN python3.10 -c "from huggingface_hub import hf_hub_download; \
        hf_hub_download('Ruicheng/moge-2-vitl', 'model.pt', \
                        revision='39c4d5e957afe587e04eec59dc2bcc3be5ecd968', local_dir='/opt/moge')" \
    && echo '3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5  /opt/moge/model.pt' | sha256sum --check \
    && rm -rf /opt/moge/.cache

# Import just the Torch UV atlas from Sprint3D, pinned with its licensing files.
ARG SPRINT3D_UV_REF=1053b33b4564c7d28a9afe31b711ba660c210204
RUN git init -q /tmp/sprint3d-uv \
    && git -C /tmp/sprint3d-uv fetch --depth 1 https://github.com/AlayaLab/sprint3d.git "$SPRINT3D_UV_REF" \
    && INSTALL_DIR="$(python3.10 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
    && mkdir -p "$INSTALL_DIR" /usr/share/doc/sprint3d-uv \
    && git -C /tmp/sprint3d-uv show FETCH_HEAD:src/sprint3d/mesh/uv_atlas.py > "$INSTALL_DIR/sprint3d_uv.py" \
    && git -C /tmp/sprint3d-uv show FETCH_HEAD:LICENSE > /usr/share/doc/sprint3d-uv/LICENSE \
    && git -C /tmp/sprint3d-uv show FETCH_HEAD:NOTICE > /usr/share/doc/sprint3d-uv/NOTICE \
    && rm -rf /tmp/sprint3d-uv

COPY rp_handler.py handler.py inference_worker.py /app/
COPY scripts /app/scripts
COPY patches/pixal3d-shared-dinov3.patch /tmp/pixal3d-shared-dinov3.patch
COPY patches/pixal3d-profile-weights.patch /tmp/pixal3d-profile-weights.patch
COPY patches/pixal3d-cascade-options.patch /tmp/pixal3d-cascade-options.patch
COPY patches/ovoxel-glb.patch /tmp/ovoxel-glb.patch
COPY patches/pixal3d-gguf.patch /tmp/pixal3d-gguf.patch

RUN python3.10 -m pip install 'gguf==0.18.0' 'safetensors==0.8.0'

# Runpod's image builder has no GPU. Importing o_voxel or the Pixal3D
# pipeline initializes FlexGEMM's Triton driver during import. Check that
# those modules are present here and load them on the GPU worker at startup.
RUN git -C /app/Pixal3D apply /tmp/pixal3d-shared-dinov3.patch \
    && git -C /app/Pixal3D apply /tmp/pixal3d-profile-weights.patch \
    && git -C /app/Pixal3D apply /tmp/pixal3d-cascade-options.patch \
    && OVOXEL_PARENT="$(python3.10 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("o_voxel").origin).parent.parent)')" \
    && git -C "$OVOXEL_PARENT" apply /tmp/ovoxel-glb.patch \
    && git -C /app/Pixal3D apply /tmp/pixal3d-gguf.patch \
    && rm /tmp/pixal3d-shared-dinov3.patch /tmp/pixal3d-profile-weights.patch /tmp/pixal3d-cascade-options.patch /tmp/ovoxel-glb.patch /tmp/pixal3d-gguf.patch \
    && python3.10 -m pip check \
    && PYTHONPATH=/app/Pixal3D python3.10 -c "from importlib.util import find_spec; \
        import torch, runpod, boto3, gguf, flash_attn, flash_attn_interface, natten; \
        from moge.model.v2 import MoGeModel; \
        from transformers import DINOv3ViTModel; \
        from sprint3d_uv import AtlasConfig, build_atlas, segment_charts; \
        from inspect import signature; from safetensors.torch import load_file; \
        assert 'backend' in signature(load_file).parameters; \
        missing = [name for name in ('o_voxel', 'flex_gemm', 'pixal3d') if find_spec(name) is None]; \
        assert not missing, f'Missing worker modules: {missing}'; \
        assert torch.version.cuda == '12.4', f'Unexpected PyTorch CUDA version: {torch.version.cuda}'; \
        assert natten.HAS_LIBNATTEN, 'NATTEN native CUDA extension could not load'" \
    && PYTHONPATH=/opt/natten-hopper python3.10 -c "import natten; \
        assert natten.__file__.startswith('/opt/natten-hopper/') and natten.HAS_LIBNATTEN" \
    && python3.10 -c "from handler import handler; assert 'error' in handler({'input': {}})"

CMD ["python3", "-u", "-m", "scripts.worker_diagnostics"]
