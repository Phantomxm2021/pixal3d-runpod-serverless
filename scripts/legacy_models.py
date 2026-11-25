"""One-time retirement of known Global Volume files after a GGUF result exists."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from scripts.asset_storage import GLB_MIME_TYPE, storage_client
from scripts.gguf_models import MODEL_FORMAT, REVISION
from scripts.model_cache import GLOBAL_MODEL_ROOT, MOGE_MODEL_NAME, RUNPOD_VOLUME, use_global_volume

LOGGER = logging.getLogger(__name__)
HI3D_FILES = (
    "hi3d/model_index.json", "hi3d/scheduler/scheduler_config.json",
    "hi3d/transformer_512/config.json", "hi3d/transformer_512/model.safetensors",
    "hi3d/transformer_1024/config.json", "hi3d/transformer_1024/model.safetensors",
    "hi3d/vae/config.json", "hi3d/vae/model.safetensors", "dinov2/config.json", "dinov2/model.safetensors",
    "clip/config.json", "clip/model.safetensors", "hi3d/LICENSE", "hi3d/NOTICE", "hi3d/AUTHORS",
    "hi3d/THIRD_PARTY_LICENSES.md", "hi3d/sparc3d/ops/flexicubes/LICENSE.txt", "manifest.json",
)
PIXAL3D_FILES = (
    "ss_dec_conv3d_16l8_fp16", "ss_flow_img_dit_1_3B_64_bf16", "shape_dec_next_dc_f16c32_fp16",
    "slat_flow_img2shape_dit_1_3B_512_bf16", "slat_flow_img2shape_dit_1_3B_1024_bf16",
    "tex_dec_next_dc_f16c32_fp16", "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
)


def _legacy_files() -> dict[Path, int]:
    hi3d = RUNPOD_VOLUME / "hi3d-models-v1"
    pixal = GLOBAL_MODEL_ROOT / "pixal3d"
    candidates = [hi3d / name for name in HI3D_FILES]
    candidates += [pixal / "pipeline.json"]
    candidates += [
        pixal / "ckpts" / f"{name}.{extension}" for name in PIXAL3D_FILES for extension in ("json", "safetensors")
    ]
    if os.environ.get("MOGE_MODEL_PATH", MOGE_MODEL_NAME) == MOGE_MODEL_NAME and Path("/opt/moge/model.pt").is_file():
        candidates.append(GLOBAL_MODEL_ROOT / "moge/model.pt")
    files = {}
    for path in candidates:
        if path.resolve() != path:
            raise ValueError(f"Legacy cache path is a link: {path}")
        if path.is_file():
            files[path] = path.stat().st_size
        elif path.exists():
            raise ValueError(f"Legacy cache path is not a regular file: {path}")
    return files


def retire_legacy_model_cache(inputs: dict) -> dict:
    key = inputs.get("validated_glb_key")
    if not isinstance(key, str) or not re.fullmatch(r"pixal3d/[0-9a-f]{32}\.glb", key):
        return {"error": "Provide validated_glb_key for the successful GGUF test."}
    bucket = os.environ.get("PIXAL3D_OUTPUT_BUCKET")
    if not bucket or not use_global_volume() or not RUNPOD_VOLUME.is_mount():
        return {"error": "The output bucket and mounted Global Volume are required."}

    try:
        proof = storage_client().head_object(Bucket=bucket, Key=key)
        metadata = proof.get("Metadata", {})
        if proof.get("ContentType") != GLB_MIME_TYPE or not proof.get("ContentLength") or (
            metadata.get("model-format"), metadata.get("model-revision")
        ) != (MODEL_FORMAT, REVISION):
            return {"error": "The saved GLB was not produced by this GGUF release."}
        files = _legacy_files()
        if inputs.get("apply") is not True:
            return {"status": "ready", "files": len(files), "model_bytes": sum(files.values())}
        for path in files:
            path.unlink(missing_ok=True)
        directories = {parent for path in files for parent in path.parents if RUNPOD_VOLUME in parent.parents}
        for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        return {"status": "retired", "files": len(files), "model_bytes": sum(files.values())}
    except Exception:
        LOGGER.exception("Retiring legacy model files failed")
        return {"error": "Could not finish retiring legacy model files. See worker logs."}
