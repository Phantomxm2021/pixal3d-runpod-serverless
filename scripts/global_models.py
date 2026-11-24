"""Publish regular model files to a Runpod Global Volume and verify the completed set."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.model_cache import (
    DINO_MODEL_NAME,
    GLOBAL_MODEL_ROOT,
    MOGE_MODEL_NAME,
    NAF_CHECKPOINT,
    NAF_CHECKPOINT_URL,
    PIXAL3D_MODEL_NAME,
    RUNPOD_VOLUME,
    configure_cache,
    pipeline_checkpoint_files,
)


DINO_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
MANIFEST = "manifest.json"


def verify_global_models(root: Path, *, include_pipeline: bool = True) -> dict[str, int]:
    try:
        sizes = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        expected = {"moge/model.pt", NAF_CHECKPOINT, *(f"dinov3/{name}" for name in DINO_FILES)}
        if not isinstance(sizes, dict):
            raise ValueError("Model manifest must be an object.")
        if include_pipeline:
            pipeline = json.loads((root / "pixal3d/pipeline.json").read_text(encoding="utf-8"))
            expected.add("pixal3d/pipeline.json")
            expected.update(f"pixal3d/{name}" for name in pipeline_checkpoint_files(pipeline))
        if include_pipeline and set(sizes) != expected:
            raise ValueError("Model manifest lists an unexpected set of files.")
        verified = {}
        for name in sorted(expected):
            size = sizes.get(name)
            path = root / name
            if type(size) is not int or size <= 0 or not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"Model file is missing or incomplete: {name}")
            verified[name] = size
        return verified
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Global Volume model files are incomplete. Run prefill_models first: {exc}") from exc


def _stream_file(url: str, destination: Path, *, size: int | None, headers: dict | None = None) -> int:
    import requests

    if size is not None and destination.is_file() and destination.stat().st_size == size:
        return size
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prefill] downloading {destination.name}", flush=True)
    written = 0
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
            if not response.ok:
                raise RuntimeError(f"Cannot download {destination.name}: HTTP {response.status_code}")
            if size is None and response.headers.get("Content-Length"):
                size = int(response.headers["Content-Length"])
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    written += output.write(chunk)
    except requests.RequestException:
        raise RuntimeError(f"Download interrupted for {destination.name}") from None
    if not written or (size is not None and written != size):
        raise RuntimeError(f"Incomplete download for {destination.name}: {written} bytes")
    return written


def prefill_global_models() -> dict:
    if not RUNPOD_VOLUME.is_mount():
        raise RuntimeError(f"Attach a Global Volume at {RUNPOD_VOLUME} before running prefill_models.")
    configure_cache()
    root = GLOBAL_MODEL_ROOT
    try:
        sizes = verify_global_models(root)
    except RuntimeError:
        sizes = _download_models(root)
    return {"status": "ready", "files": len(sizes), "model_bytes": sum(sizes.values())}


def _download_models(root: Path) -> dict[str, int]:
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    api = HfApi()
    headers = build_hf_headers()
    sizes = {}
    for folder, repo, files in (
        ("pixal3d", PIXAL3D_MODEL_NAME, ("pipeline.json",)),
        ("moge", MOGE_MODEL_NAME, ("model.pt",)),
        ("dinov3", DINO_MODEL_NAME, DINO_FILES),
    ):
        info = api.model_info(repo, files_metadata=True)
        available = {entry.rfilename: entry.size for entry in info.siblings}

        def download_files(names):
            for name in names:
                if name not in available or not available[name]:
                    raise RuntimeError(f"Cannot determine file size for {repo}/{name}")
                relative = f"{folder}/{name}"
                sizes[relative] = _stream_file(
                    hf_hub_url(repo, name, revision=info.sha), root / relative,
                    size=available[name], headers=headers,
                )

        download_files(files)
        if folder == "pixal3d":
            config = json.loads((root / "pixal3d/pipeline.json").read_text(encoding="utf-8"))
            download_files(pipeline_checkpoint_files(config))

    sizes[NAF_CHECKPOINT] = _stream_file(NAF_CHECKPOINT_URL, root / NAF_CHECKPOINT, size=None)
    # Publish the manifest only after every file closes successfully. A retry
    # can reuse finished Hugging Face files without writing symlinks or locks.
    (root / MANIFEST).write_text(json.dumps(sizes), encoding="utf-8")
    return verify_global_models(root)
