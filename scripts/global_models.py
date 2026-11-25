"""Publish regular model files to a Runpod Global Volume and verify the completed set."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from scripts.model_cache import (
    DINO_MODEL_NAME,
    GLOBAL_MODEL_ROOT,
    NAF_CHECKPOINT,
    NAF_CHECKPOINT_URL,
    RUNPOD_VOLUME,
    configure_cache,
)


DINO_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
MANIFEST = "manifest.json"


@contextmanager
def stage_dinov3(source: Path, scratch: Path) -> Iterator[Path]:
    """Keep network-backed reads out of the model's mmap and CUDA copy."""
    with tempfile.TemporaryDirectory(prefix="dinov3-", dir=scratch) as directory:
        staged = Path(directory)
        for name in DINO_FILES:
            shutil.copyfile(source / name, staged / name)
        yield staged


def verify_global_models(root: Path) -> dict[str, int]:
    try:
        sizes = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        expected = {NAF_CHECKPOINT, *(f"dinov3/{name}" for name in DINO_FILES)}
        if not isinstance(sizes, dict):
            raise ValueError("Model manifest must be an object.")
        verified = {}
        for name in sorted(expected):
            size = sizes.get(name)
            path = root / name
            if type(size) is not int or size <= 0 or not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"Model file is missing or incomplete: {name}")
            verified[name] = size
        return verified
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Global Volume model files are incomplete. Run scripts.prefetch_models on the attached volume first: {exc}") from exc


def _stream_file(
    url: str, destination: Path, *, size: int | None, headers: dict | None = None, sha256: str | None = None,
) -> int:
    import requests

    if sha256 is None and size is not None and destination.is_file() and destination.stat().st_size == size:
        return size
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prefill] downloading {destination.name}", flush=True)
    written = 0
    digest = hashlib.sha256() if sha256 is not None else None
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
            if not response.ok:
                raise RuntimeError(f"Cannot download {destination.name}: HTTP {response.status_code}")
            if size is None and response.headers.get("Content-Length"):
                size = int(response.headers["Content-Length"])
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    written += output.write(chunk)
                    if digest is not None:
                        digest.update(chunk)
    except requests.RequestException:
        raise RuntimeError(f"Download interrupted for {destination.name}") from None
    if not written or (size is not None and written != size):
        raise RuntimeError(f"Incomplete download for {destination.name}: {written} bytes")
    if digest is not None and digest.hexdigest() != sha256:
        raise RuntimeError(f"Unexpected SHA-256 for {destination.name}")
    return written


def prefill_global_models() -> dict:
    if not RUNPOD_VOLUME.is_mount():
        raise RuntimeError(f"Attach a Global Volume at {RUNPOD_VOLUME} before preparing auxiliary models.")
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
    repo = DINO_MODEL_NAME
    info = api.model_info(repo, files_metadata=True, timeout=30)
    available = {entry.rfilename: entry.size for entry in info.siblings}
    sizes = {}
    for name in DINO_FILES:
        if not available.get(name):
            raise RuntimeError(f"Cannot determine file size for {repo}/{name}")
        relative = f"dinov3/{name}"
        sizes[relative] = _stream_file(
            hf_hub_url(repo, name, revision=info.sha), root / relative,
            size=available[name], headers=headers,
        )

    sizes[NAF_CHECKPOINT] = _stream_file(NAF_CHECKPOINT_URL, root / NAF_CHECKPOINT, size=None)
    (root / MANIFEST).write_text(json.dumps(sizes), encoding="utf-8")
    return verify_global_models(root)
