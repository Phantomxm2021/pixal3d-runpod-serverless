"""Export stored Pixal3D source assets with headless Blender."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from scripts.asset_storage import GLB_MIME_TYPE, asset_source_key, download_url, storage_client
from scripts.mesh_stats import glb_stats

LOGGER = logging.getLogger(__name__)
EXPORT_SCRIPT = Path(__file__).resolve().parent / "scripts" / "blender_export.py"
EXPORT_VERSION = "blender-5.2.1-v1"
MAX_SOURCE_BYTES = 250_000_000
MISSING_OBJECT = {"404", "NoSuchKey", "NotFound"}


def _s3_error_code(error: Exception) -> str | None:
    return getattr(error, "response", {}).get("Error", {}).get("Code")


def export_glb(inputs: dict) -> dict:
    asset_id = inputs.get("asset_id")
    if not isinstance(asset_id, str) or not re.fullmatch(r"[0-9a-f]{32}", asset_id):
        return {"error": "Use the asset_id from the USDZ generation response."}
    triangle_limit = inputs.get("target_triangles", 30_000)
    texture_limit = inputs.get("texture_size", 2048)
    if type(triangle_limit) is not int or not 1_000 <= triangle_limit <= 200_000:
        return {"error": "input.target_triangles must be an integer between 1000 and 200000."}
    if type(texture_limit) is not int or texture_limit not in (512, 1024, 2048):
        return {"error": "input.texture_size must be 512, 1024 or 2048."}
    bucket = os.environ.get("PIXAL3D_OUTPUT_BUCKET")
    if not bucket:
        return {"error": "Set PIXAL3D_OUTPUT_BUCKET on the endpoint."}

    variant = f"{EXPORT_VERSION}-{triangle_limit}-{texture_limit}"
    key = f"pixal3d/exports/{asset_id}/{variant}.glb"
    started = time.monotonic()
    timings = {}
    try:
        storage = storage_client()
        try:
            cached = storage.head_object(Bucket=bucket, Key=key)
        except Exception as error:
            if _s3_error_code(error) not in MISSING_OBJECT:
                raise
            cached = None

        if cached:
            stats = {
                "triangle_count": int(cached["Metadata"]["triangle-count"]),
                "vertex_count": int(cached["Metadata"]["vertex-count"]),
                "size_bytes": cached["ContentLength"],
            }
        else:
            try:
                source = storage.get_object(Bucket=bucket, Key=asset_source_key(asset_id))
            except Exception as error:
                if _s3_error_code(error) in MISSING_OBJECT:
                    return {"error": "No USDZ source was found for this asset_id."}
                raise

            with tempfile.TemporaryDirectory(prefix="pixal3d-export-") as temp, source["Body"] as body:
                if not 0 < source["ContentLength"] <= MAX_SOURCE_BYTES:
                    return {"error": "The source USDZ exceeds the export size limit or is empty."}
                source_path = Path(temp) / "source.usdz"
                with source_path.open("wb") as output:
                    shutil.copyfileobj(body, output, length=1024 * 1024)
                if source_path.stat().st_size != source["ContentLength"]:
                    raise ValueError("Incomplete USDZ download.")
                timings["download_seconds"] = round(time.monotonic() - started, 2)

                glb = Path(temp) / "output.glb"
                stage = time.monotonic()
                subprocess.run([
                    os.environ.get("BLENDER_BINARY", "blender"), "-b", "-t", "4",
                    "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
                    "--python", str(EXPORT_SCRIPT), "--", str(source_path), str(glb),
                    str(triangle_limit), str(texture_limit),
                ], check=True, timeout=240)
                stats = glb_stats(glb)
                timings["blender_seconds"] = round(time.monotonic() - stage, 2)

                stage = time.monotonic()
                storage.upload_file(str(glb), bucket, key, ExtraArgs={
                    "ContentType": GLB_MIME_TYPE,
                    "Metadata": {
                        "triangle-count": str(stats["triangle_count"]),
                        "vertex-count": str(stats["vertex_count"]),
                    },
                })
                timings["upload_seconds"] = round(time.monotonic() - stage, 2)

        return {
            "asset_id": asset_id, "glb_url": download_url(storage, bucket, key),
            "target_triangles": triangle_limit, "texture_size": texture_limit,
            "cached": cached is not None, **stats,
            "timing": {"export_seconds": round(time.monotonic() - started, 2), **timings},
        }
    except subprocess.TimeoutExpired:
        LOGGER.exception("Blender exceeded its 240-second export limit")
        return {"error": "Blender timed out while exporting this asset."}
    except Exception:
        LOGGER.exception("USDZ to GLB export failed")
        return {"error": "USDZ to GLB export failed. See the worker logs."}
