"""Shared cache settings for the worker and the volume-prefetch script."""

import json
import os
import re
from pathlib import Path


PIXAL3D_MODEL_NAME = "TencentARC/Pixal3D"
MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
DINO_MODEL_NAME = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_BIREFNET_MODEL = "ZhengPeng7/BiRefNet"
RUNPOD_VOLUME = Path("/runpod-volume")
GLOBAL_MODEL_ROOT = RUNPOD_VOLUME / "pixal3d-models-v1"
RUNPOD_MODEL_HUB = RUNPOD_VOLUME / "huggingface-cache" / "hub"
NAF_CHECKPOINT = "naf_release.pth"
NAF_CHECKPOINT_URL = f"https://github.com/valeoai/NAF/releases/download/model/{NAF_CHECKPOINT}"


def configure_cache() -> None:
    volume = RUNPOD_VOLUME
    if use_global_volume() or use_runpod_model_cache():
        # Keep writable caches on the container filesystem when using prefetched models.
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
        return
    hf_home = Path(os.environ.setdefault("HF_HOME", str(volume / "huggingface")))
    os.environ.setdefault("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))
    os.environ.setdefault("TORCH_HOME", str(volume / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(volume / "cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(volume / "triton"))
    os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(volume / "flex_gemm_autotune.json"))


def use_global_volume() -> bool:
    return os.environ.get("PIXAL3D_GLOBAL_VOLUME") == "1"


def use_runpod_model_cache() -> bool:
    return os.environ.get("PIXAL3D_RUNPOD_MODEL_CACHE") == "1"


def pipeline_checkpoint_files(config: dict) -> list[str]:
    return [
        f"{checkpoint}.{extension}"
        for checkpoint in config["args"]["models"].values()
        for extension in ("json", "safetensors")
    ]


def _runpod_revision(repo: Path) -> str:
    pinned = os.environ.get("PIXAL3D_MODEL_REVISION")
    if pinned is not None:
        return pinned
    try:
        return (repo / "refs/main").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        revisions = [path.name for path in (repo / "snapshots").iterdir() if path.is_dir()]
        if len(revisions) != 1:
            raise ValueError("Expected one snapshot when refs/main is unavailable.")
        return revisions[0]


def runpod_pipeline_snapshot(*, repair: bool = False) -> tuple[Path, dict]:
    repo = RUNPOD_MODEL_HUB / f"models--{PIXAL3D_MODEL_NAME.replace('/', '--')}"
    try:
        pinned = "PIXAL3D_MODEL_REVISION" in os.environ
        if repair and not pinned:
            raise ValueError("Set PIXAL3D_MODEL_REVISION before repairing the node cache.")
        revision = _runpod_revision(repo)
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError(f"Invalid snapshot revision: {revision!r}")

        snapshot = repo / "snapshots" / revision
        sizes = None
        if pinned:
            from huggingface_hub import HfApi

            info = HfApi().model_info(PIXAL3D_MODEL_NAME, revision=revision, files_metadata=True, timeout=15)
            if info.sha != revision:
                raise ValueError(f"Hugging Face returned an unexpected revision: {info.sha!r}")
            sizes = {entry.rfilename: entry.size for entry in info.siblings}

        def check_file(filename: str) -> Path:
            path = snapshot / filename
            expected = sizes.get(filename) if sizes is not None else None
            if sizes is not None and (type(expected) is not int or expected <= 0):
                raise ValueError(f"No valid size in revision {revision} for {filename}")
            present = path.is_file()
            actual = path.stat().st_size if present else 0
            if repair and actual != expected:
                from huggingface_hub import hf_hub_download

                print(f"[cache] Repairing {revision}/{filename}", flush=True)
                hf_hub_download(PIXAL3D_MODEL_NAME, filename, revision=revision,
                                cache_dir=RUNPOD_MODEL_HUB, force_download=present)
                actual = path.stat().st_size if path.is_file() else 0
            if actual <= 0 or (expected is not None and actual != expected):
                raise ValueError(f"Incomplete {snapshot / filename}: {actual} bytes, expected {expected or 'nonempty'}")
            return path

        config = json.loads(check_file("pipeline.json").read_text(encoding="utf-8"))
        for filename in pipeline_checkpoint_files(config):
            check_file(filename)
        return snapshot, config
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise RuntimeError(f"Runpod node model cache is missing or incomplete at {repo}: {exc}") from exc


def pipeline_snapshot(model_path: str) -> tuple[Path, dict]:
    config_path = Path(model_path).resolve() / "pipeline.json"
    if config_path.is_file():
        return config_path.parent, json.loads(config_path.read_text(encoding="utf-8"))

    from huggingface_hub import hf_hub_download, snapshot_download

    config_path = Path(hf_hub_download(model_path, "pipeline.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # The repository also contains multi-view weights. Download only the files
    # referenced by the single-image pipeline, at the same revision as its config.
    checkpoint_files = pipeline_checkpoint_files(config)
    if all((config_path.parent / filename).is_file() for filename in checkpoint_files):
        return config_path.parent, config
    snapshot = snapshot_download(
        model_path, revision=config_path.parent.name, allow_patterns=checkpoint_files
    )
    return Path(snapshot), config
