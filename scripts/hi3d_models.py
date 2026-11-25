"""Prepare and verify the fixed Hi3D artifacts on the existing Global Volume."""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.global_models import _stream_file
from scripts.model_cache import RUNPOD_VOLUME

HI3D_REVISION = "705fc0901c5d896cdcbf1dbf1feaa173e62bdd3a"
MODEL_SOURCES = {
    "hi3d": (
        "hi3d-official/Hi3Dv1.0", HI3D_REVISION, (
            "model_index.json", "scheduler/scheduler_config.json",
            "transformer_512/config.json", "transformer_512/model.safetensors",
            "transformer_1024/config.json", "transformer_1024/model.safetensors",
            "vae/config.json", "vae/model.safetensors",
            "LICENSE", "NOTICE", "AUTHORS", "THIRD_PARTY_LICENSES.md", "sparc3d/ops/flexicubes/LICENSE.txt",
        ),
    ),
    "dinov2": (
        "facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c", ("config.json", "model.safetensors"),
    ),
    "clip": (
        "openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41", ("config.json", "model.safetensors"),
    ),
}
EXPECTED_FILES = {
    f"{folder}/{name}" for folder, (_, _, files) in MODEL_SOURCES.items() for name in files
}


def hi3d_model_root() -> Path:
    return Path(os.environ.get("HI3D_MODEL_ROOT", str(RUNPOD_VOLUME / "hi3d-models-v1")))


def verify_hi3d_models(root: Path) -> dict[str, int]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("revisions") != {folder: source[1] for folder, source in MODEL_SOURCES.items()}:
        raise ValueError("Unexpected Hi3D model revisions.")
    files = manifest["files"]
    if set(files) != EXPECTED_FILES:
        raise ValueError("Hi3D model manifest does not list the required files.")
    for name, size in files.items():
        if type(size) is not int or size <= 0 or (root / name).stat().st_size != size:
            raise ValueError(f"Invalid size for {name}.")
    return files


def _download_file(url: str, destination: Path, size: int, headers: dict) -> int:
    if destination.is_file() and destination.stat().st_size == size:
        return size
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as file:
        temporary_path = Path(file.name)
    try:
        _stream_file(url, temporary_path, size=size, headers=headers)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return size


def prefill_hi3d_models() -> dict:
    if "HI3D_MODEL_ROOT" not in os.environ and not RUNPOD_VOLUME.is_mount():
        raise RuntimeError(f"Attach the Global Volume at {RUNPOD_VOLUME} first.")
    root = hi3d_model_root()
    try:
        files = verify_hi3d_models(root)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        files = _prefill(root)
    return {"status": "ready", "hi3d_revision": HI3D_REVISION, "files": len(files), "model_bytes": sum(files.values())}


def _prefill(root: Path) -> dict[str, int]:
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    api = HfApi()
    headers = build_hf_headers()
    downloads = {}
    for folder, (repo, revision, names) in MODEL_SOURCES.items():
        info = api.model_info(repo, revision=revision, files_metadata=True, timeout=30)
        if info.sha != revision:
            raise RuntimeError(f"Unexpected Hugging Face revision for {repo}.")
        sizes = {entry.rfilename: entry.size for entry in info.siblings}
        for name in names:
            size = sizes.get(name)
            if type(size) is not int or size <= 0:
                raise RuntimeError(f"No file size available for {repo}/{name}.")
            downloads[f"{folder}/{name}"] = (hf_hub_url(repo, name, revision=revision), size)

    def download(item: tuple[str, tuple[str, int]]) -> tuple[str, int]:
        name, (url, size) = item
        if not (root / name).is_file() or (root / name).stat().st_size != size:
            print(f"[hi3d prefill] Downloading {name} ({size} bytes)", flush=True)
        return name, _download_file(url, root / name, size, headers)

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = dict(pool.map(download, downloads.items()))
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, prefix=".manifest-", delete=False) as file:
        manifest = Path(file.name)
        try:
            json.dump({"revisions": {name: source[1] for name, source in MODEL_SOURCES.items()}, "files": files}, file)
            file.close()
            manifest.replace(root / "manifest.json")
        finally:
            manifest.unlink(missing_ok=True)
    return verify_hi3d_models(root)
