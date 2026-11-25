"""Refine a stored Pixal3D white mesh in the worker's isolated Hi3D runtime."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from scripts.asset_storage import GLB_MIME_TYPE, download_url, s3_error_code, storage_client, white_mesh_key
from scripts.hi3d_models import HI3D_REVISION, hi3d_model_root, verify_hi3d_models
from scripts.mesh_options import triangle_budget
from scripts.mesh_stats import glb_stats

LOGGER = logging.getLogger(__name__)
WORKER_ROOT = Path(__file__).resolve().parent
MAX_WHITE_MESH_BYTES = 100_000_000
HI3D_TIMEOUT_SECONDS = 480


@dataclass(frozen=True)
class RefinementOptions:
    asset_id: str
    resolution: int
    num_inference_steps: int
    target_triangles: int | None


def refinement_options(inputs: dict) -> RefinementOptions:
    asset_id = inputs.get("asset_id")
    if not isinstance(asset_id, str) or not re.fullmatch(r"[0-9a-f]{32}", asset_id):
        raise ValueError("Use the asset_id from a white mesh response saved with URL output.")
    resolution = inputs.get("resolution", 512)
    if type(resolution) is not int or resolution not in (512, 1024):
        raise ValueError("Hi3D input.resolution must be 512 or 1024.")
    steps = inputs.get("num_inference_steps", 25)
    if type(steps) is not int or not 1 <= steps <= 50:
        raise ValueError("input.num_inference_steps must be an integer between 1 and 50.")
    target = None
    if "target_triangles" in inputs:
        target = triangle_budget(inputs, default_triangles=200_000, max_triangles=1_000_000)
    return RefinementOptions(asset_id, resolution, steps, target)


def refine_white_mesh(
    input_image: Path, directory: Path, bucket: str, options: RefinementOptions, *, seed: int, skip_rembg: bool,
) -> dict:
    python = Path(os.environ.get("HI3D_PYTHON", "/opt/hi3d-runtime/bin/python"))
    root = hi3d_model_root()
    if not python.is_file():
        return {"error": "The Hi3D runtime is missing from this worker image."}
    try:
        verify_hi3d_models(root)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        LOGGER.warning("Hi3D cache verification failed: %s", error)
        return {"error": "Hi3D model files are incomplete. Run prepare_hi3d_cache on this endpoint first."}

    started = time.monotonic()
    timings = {}
    try:
        storage = storage_client()
        try:
            source = storage.get_object(Bucket=bucket, Key=white_mesh_key(options.asset_id))
        except Exception as error:
            if s3_error_code(error) in {"404", "NoSuchKey", "NotFound"}:
                return {"error": "No saved white mesh was found for this asset_id."}
            raise
        source_path = directory / "coarse.glb"
        with source["Body"] as body:
            size = source["ContentLength"]
            if not 0 < size <= MAX_WHITE_MESH_BYTES:
                return {"error": "The saved white mesh is empty or exceeds 100 MB."}
            with source_path.open("wb") as file:
                shutil.copyfileobj(body, file, length=1024 * 1024)
        if source_path.stat().st_size != size:
            raise ValueError("Incomplete white mesh download.")
        source_stats = glb_stats(source_path)
        timings["source_download_seconds"] = round(time.monotonic() - started, 2)

        inference = sys.modules.get("inference_worker")
        if inference is not None:
            stage = time.monotonic()
            inference.release_models()
            timings["pixal3d_release_seconds"] = round(time.monotonic() - stage, 2)

        output = directory / "refined.glb"
        report = directory / "hi3d-report.json"
        command = [
            str(python), "-u", "-m", "scripts.hi3d_worker", "--image", str(input_image),
            "--source", str(source_path), "--output", str(output), "--report", str(report),
            "--model-root", str(root), f"--seed={seed}", "--resolution", str(options.resolution),
            "--steps", str(options.num_inference_steps),
        ]
        if options.target_triangles is not None:
            command += ["--target-triangles", str(options.target_triangles)]
        if skip_rembg:
            command.append("--skip-rembg")
        stage = time.monotonic()
        subprocess.run(command, cwd=WORKER_ROOT, check=True, timeout=HI3D_TIMEOUT_SECONDS, env={
            **os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(WORKER_ROOT),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TRITON_CACHE_DIR": "/tmp/hi3d-cache/triton", "U2NET_HOME": "/opt/hi3d-rembg",
        })
        timings["hi3d_process_seconds"] = round(time.monotonic() - stage, 2)
        metrics = json.loads(report.read_text(encoding="utf-8"))
        timings.update(metrics.pop("timing"))
        stats = glb_stats(output)

        stage = time.monotonic()
        asset_id = uuid.uuid4().hex
        key = white_mesh_key(asset_id)
        storage.upload_file(str(output), bucket, key, ExtraArgs={"ContentType": GLB_MIME_TYPE})
        url = download_url(storage, bucket, key)
        timings["upload_seconds"] = round(time.monotonic() - stage, 2)
        return {
            "asset_type": "white_mesh", "asset_id": asset_id, "source_asset_id": options.asset_id, "glb_url": url,
            "hi3d_revision": HI3D_REVISION, "seed": seed, "target_triangles": options.target_triangles,
            "source_triangle_count": source_stats["triangle_count"], **metrics, **stats, "timing": timings,
        }
    except subprocess.TimeoutExpired:
        LOGGER.exception("Hi3D exceeded the %s-second refinement limit", HI3D_TIMEOUT_SECONDS)
        return {"error": f"Hi3D timed out after {HI3D_TIMEOUT_SECONDS} seconds."}
    except Exception:
        LOGGER.exception("Hi3D refinement failed")
        return {"error": "Hi3D refinement failed. See the worker logs for details."}
