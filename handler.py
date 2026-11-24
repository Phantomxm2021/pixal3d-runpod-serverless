"""Runpod queue handler for Pixal3D image-to-GLB jobs."""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)
GLB_MIME_TYPE = "model/gltf-binary"
MAX_IMAGE_BYTES = 20_000_000
# Leave room for base64 expansion and the JSON envelope inside /run's 10 MB limit.
MAX_INLINE_GLB_BYTES = 7_000_000


def _download_image(url: str, output: Path) -> None:
    request = Request(url, headers={"User-Agent": "Runpod-Pixal3D-Worker"})
    try:
        with urlopen(request, timeout=60) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
    except (URLError, TimeoutError) as exc:
        raise ValueError("Could not download input.image_url.") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("The downloaded image exceeds 20 MB.")
    output.write_bytes(data)


def _decode_image(data: str, output: Path) -> None:
    data = data.strip()
    if data.startswith("data:"):
        header, _, data = data.partition(",")
        if not header.endswith(";base64"):
            raise ValueError("input.image_base64 must contain base64 image data.")
    try:
        decoded = base64.b64decode("".join(data.split()), validate=True)
    except binascii.Error as exc:
        raise ValueError("input.image_base64 must contain valid base64.") from exc
    if not decoded or len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("The decoded image must be between 1 byte and 20 MB.")
    output.write_bytes(decoded)


def _upload_glb(path: Path, bucket: str) -> str:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("PIXAL3D_OUTPUT_ENDPOINT_URL"),
        config=Config(signature_version="s3v4"),
    )
    key = f"pixal3d/{uuid.uuid4().hex}.glb"
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": GLB_MIME_TYPE})
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600
    )


def _run_inference(input_path: Path, output_path: Path, *, seed: int, skip_rembg: bool) -> dict:
    from inference_worker import run_inference

    return run_inference(input_path, output_path, seed=seed, skip_rembg=skip_rembg)


def handler(job: dict) -> dict:
    """Generate a GLB and return it inline or as a temporary download URL."""
    inputs = job.get("input")
    if not isinstance(inputs, dict):
        return {"error": "Provide an input object with image_url or image_base64."}

    image_url = inputs.get("image_url")
    image_base64 = inputs.get("image_base64")
    if not image_url and not image_base64:
        return {"error": "Provide input.image_url or input.image_base64."}
    if image_url:
        try:
            url = urlsplit(image_url) if isinstance(image_url, str) else None
            if not url or url.scheme not in {"http", "https"} or not url.hostname:
                raise ValueError
        except ValueError:
            return {"error": "input.image_url must be an HTTP or HTTPS URL."}
    elif not isinstance(image_base64, str):
        return {"error": "input.image_base64 must be a string."}

    suffix = inputs.get("suffix", ".png")
    if not isinstance(suffix, str) or not re.fullmatch(r"\.?[a-zA-Z0-9]{1,10}", suffix):
        return {"error": "input.suffix must be a file extension such as .png."}
    seed = inputs.get("seed", 42)
    try:
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise ValueError
        seed = int(seed)
        if not -(2**63) <= seed < 2**64:
            raise ValueError
    except ValueError:
        return {"error": "input.seed must be an integer in the PyTorch seed range."}
    skip_rembg = inputs.get("skip_rembg", False)
    if not isinstance(skip_rembg, bool):
        return {"error": "input.skip_rembg must be a boolean."}

    output_format = inputs.get("output_format", "base64")
    if output_format not in ("base64", "url"):
        return {"error": "input.output_format must be base64 or url."}
    bucket = os.environ.get("PIXAL3D_OUTPUT_BUCKET")
    if output_format == "url" and not bucket:
        return {"error": "Set PIXAL3D_OUTPUT_BUCKET on the endpoint to use URL output."}

    with tempfile.TemporaryDirectory(prefix="pixal3d-job-") as workdir:
        directory = Path(workdir)
        input_path = directory / f"input.{suffix.lstrip('.')}"
        output_path = directory / "output.glb"
        started = time.monotonic()
        try:
            if image_url:
                _download_image(image_url, input_path)
            else:
                _decode_image(image_base64, input_path)
        except ValueError as exc:
            return {"error": str(exc)}
        fetch_seconds = round(time.monotonic() - started, 2)

        try:
            timing = _run_inference(input_path, output_path, seed=seed, skip_rembg=skip_rembg)
            size = output_path.stat().st_size if output_path.is_file() else 0
        except Exception:
            LOGGER.exception("Pixal3D inference failed")
            return {"error": "Pixal3D inference failed. See the worker logs for details."}
        if not size:
            return {"error": "Inference finished but produced no GLB."}

        result = {
            "filename": output_path.name,
            "mime_type": GLB_MIME_TYPE,
            "size_bytes": size,
            "timing": {"fetch_seconds": fetch_seconds, **timing},
        }
        started = time.monotonic()
        if output_format == "url":
            try:
                result["glb_url"] = _upload_glb(output_path, bucket)
            except Exception:
                LOGGER.exception("GLB upload failed")
                return {"error": "GLB upload failed. Check the worker's S3 configuration and logs."}
            result["timing"]["upload_seconds"] = round(time.monotonic() - started, 2)
        else:
            if size > MAX_INLINE_GLB_BYTES:
                return {"error": "GLB exceeds the 7 MB inline limit. Use input.output_format=url."}
            result["glb_base64"] = base64.b64encode(output_path.read_bytes()).decode("ascii")
            result["timing"]["encode_seconds"] = round(time.monotonic() - started, 2)
        return result
