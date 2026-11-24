"""Shared cache settings for the worker and the volume-prefetch script."""

import json
import os
from pathlib import Path


PIXAL3D_MODEL_NAME = "TencentARC/Pixal3D"
MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
DINO_MODEL_NAME = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_BIREFNET_MODEL = "ZhengPeng7/BiRefNet"


def configure_cache() -> None:
    volume = Path("/runpod-volume")
    hf_home = Path(os.environ.setdefault("HF_HOME", str(volume / "huggingface")))
    os.environ.setdefault("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))
    os.environ.setdefault("TORCH_HOME", str(volume / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(volume / "cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(volume / "triton"))
    os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(volume / "flex_gemm_autotune.json"))


def pipeline_snapshot(model_path: str) -> tuple[Path, dict]:
    config_path = Path(model_path).resolve() / "pipeline.json"
    if config_path.is_file():
        return config_path.parent, json.loads(config_path.read_text(encoding="utf-8"))

    from huggingface_hub import hf_hub_download, snapshot_download

    config_path = Path(hf_hub_download(model_path, "pipeline.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # The repository also contains multi-view weights. Download only the files
    # referenced by the single-image pipeline, at the same revision as its config.
    checkpoint_files = [
        f"{checkpoint}.{extension}"
        for checkpoint in config["args"]["models"].values()
        for extension in ("json", "safetensors")
    ]
    if all((config_path.parent / filename).is_file() for filename in checkpoint_files):
        return config_path.parent, config
    snapshot = snapshot_download(
        model_path, revision=config_path.parent.name, allow_patterns=checkpoint_files
    )
    return Path(snapshot), config
