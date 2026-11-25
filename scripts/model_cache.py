"""Shared cache settings for the worker and the volume-prefetch script."""

import os
from pathlib import Path


MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
DINO_MODEL_NAME = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_BIREFNET_MODEL = "ZhengPeng7/BiRefNet"
RUNPOD_VOLUME = Path("/runpod-volume")
GLOBAL_MODEL_ROOT = RUNPOD_VOLUME / "pixal3d-models-v1"
NAF_CHECKPOINT = "naf_release.pth"
NAF_CHECKPOINT_URL = f"https://github.com/valeoai/NAF/releases/download/model/{NAF_CHECKPOINT}"


def configure_cache() -> None:
    # Global Volume holds persistent model files. Runtime caches are writable locally.
    local = Path("/tmp/pixal3d-cache")
    hf = local / "huggingface"
    os.environ.update({
        "HF_HOME": str(hf),
        "HF_HUB_CACHE": str(hf / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf / "hub"),
        "HF_XET_CACHE": str(hf / "xet"),
        "HF_MODULES_CACHE": str(hf / "modules"),
        "TORCH_HOME": str(local / "torch"),
        "XDG_CACHE_HOME": str(local),
        "TRITON_CACHE_DIR": str(local / "triton"),
        "CUDA_CACHE_PATH": str(local / "cuda"),
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(local / "flex_gemm_autotune.json"),
    })
    local.mkdir(parents=True, exist_ok=True)


def use_global_volume() -> bool:
    return os.environ.get("PIXAL3D_GLOBAL_VOLUME") == "1"
