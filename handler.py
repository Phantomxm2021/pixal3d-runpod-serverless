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

from scripts.asset_storage import GLB_MIME_TYPE, download_url, storage_client
from scripts.generation_options import GenerationOptions, generation_options
from scripts.gguf_models import MODEL_FORMAT, REVISION

LOGGER = logging.getLogger(__name__)
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
    client = storage_client()
    key = f"pixal3d/{uuid.uuid4().hex}.glb"
    client.upload_file(str(path), bucket, key, ExtraArgs={
        "ContentType": GLB_MIME_TYPE,
        "Metadata": {"model-format": MODEL_FORMAT, "model-revision": REVISION},
    })
    return download_url(client, bucket, key)


def _run_inference(
    input_path: Path, output_path: Path, *, seed: int, skip_rembg: bool,
    options: GenerationOptions,
) -> tuple[dict[str, float], dict[str, int]]:
    print("[request] Importing inference worker", flush=True)
    started = time.monotonic()
    from inference_worker import MODEL_LOAD_TIMINGS, prepare_models, run_job_sample
    cold = prepare_models(low_vram=options.low_vram)

    worker_ready_seconds = round(time.monotonic() - started, 2)
    print(f"[request] Inference worker ready in {worker_ready_seconds:.2f}s, starting inference", flush=True)
    timing, geometry = run_job_sample(input_path, output_path, seed=seed, skip_rembg=skip_rembg, options=options)
    if cold:
        timing = {"worker_ready_seconds": worker_ready_seconds, **MODEL_LOAD_TIMINGS, **timing}
    return timing, geometry


def _deliver_asset(path: Path, result: dict, output_format: str, bucket: str | None) -> dict:
    size = result["size_bytes"]
    started = time.monotonic()
    if output_format == "url" or (output_format == "auto" and size > MAX_INLINE_GLB_BYTES and bucket):
        try:
            result["glb_url"] = _upload_glb(path, bucket)
        except Exception:
            LOGGER.exception("GLB upload failed")
            return {"error": "GLB upload failed. Check the worker's S3 configuration and logs."}
        result["timing"]["upload_seconds"] = round(time.monotonic() - started, 2)
    else:
        if size > MAX_INLINE_GLB_BYTES:
            return {"error": "GLB exceeds the 7 MB inline limit. Configure PIXAL3D_OUTPUT_BUCKET and use input.output_format=url."}
        result["glb_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
        result["timing"]["encode_seconds"] = round(time.monotonic() - started, 2)
    return result


def handler(job: dict) -> dict:
    """Generate a textured GLB on the Pixal3D endpoint."""
    inputs = job.get("input")
    if not isinstance(inputs, dict):
        return {"error": "Provide an input object with image_url or image_base64."}

    if inputs.get("action") == "retire_legacy_model_cache":
        from scripts.legacy_models import retire_legacy_model_cache

        return retire_legacy_model_cache(inputs)
    if inputs.get("action") is not None:
        return {"error": "This endpoint generates a textured GLB directly. Send input.image_url or input.image_base64 without an action."}
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

    output_format = inputs.get("output_format", "auto")
    if output_format not in ("auto", "base64", "url"):
        return {"error": "input.output_format must be auto, base64 or url."}
    bucket = os.environ.get("PIXAL3D_OUTPUT_BUCKET")
    if inputs.get("asset_format", "glb") != "glb":
        return {"error": "This endpoint outputs GLB."}
    if output_format == "url" and not bucket:
        return {"error": "Set PIXAL3D_OUTPUT_BUCKET on the endpoint to use URL output."}

    try:
        options = generation_options(inputs)
    except ValueError as exc:
        return {"error": str(exc)}
    if options.num_samples > 1:
        if output_format == "base64" or not bucket:
            return {"error": "Multiple samples require URL output and PIXAL3D_OUTPUT_BUCKET."}
        output_format = "url"

    with tempfile.TemporaryDirectory(prefix="pixal3d-job-") as workdir:
        directory = Path(workdir)
        input_path = directory / f"input.{suffix.lstrip('.')}"
        started = time.monotonic()
        print("[request] Reading input image", flush=True)
        try:
            if image_url:
                _download_image(image_url, input_path)
            else:
                _decode_image(image_base64, input_path)
        except ValueError as exc:
            return {"error": str(exc)}
        fetch_seconds = round(time.monotonic() - started, 2)
        print(f"[request] Input image ready after {fetch_seconds:.2f}s", flush=True)

        generated = []
        for index in range(options.num_samples):
            filename = "output" if options.num_samples == 1 else f"output-{index + 1}"
            output_path = directory / f"{filename}.glb"
            sample_seed = seed + index
            if sample_seed >= 2**64:
                sample_seed -= 2**64
            try:
                timing, geometry = _run_inference(
                    input_path, output_path, seed=sample_seed, skip_rembg=skip_rembg, options=options,
                )
                size = output_path.stat().st_size if output_path.is_file() else 0
            except Exception:
                LOGGER.exception("Pixal3D inference failed")
                return {"error": "Pixal3D inference failed. See the worker logs for details."}
            if not size:
                return {"error": "Inference finished but produced no GLB."}
            print(f"[request] GLB generated ({size} bytes)", flush=True)
            result = {
                "filename": output_path.name,
                "mime_type": GLB_MIME_TYPE,
                "model_format": MODEL_FORMAT,
                "model_revision": REVISION,
                "size_bytes": size,
                "seed": sample_seed,
                "pipeline_type": options.pipeline["pipeline_type"],
                "texture_size": options.postprocess["texture_size"],
                "texture_format": options.texture_format,
                "uv_backend": options.postprocess["uv_backend"],
                "target_triangles": options.postprocess["decimation_target"],
                **geometry,
                "timing": {"fetch_seconds": fetch_seconds, **timing},
            }
            generated.append((output_path, result))

        assets = []
        for path, result in generated:
            delivered = _deliver_asset(path, result, output_format, bucket)
            if "error" in delivered:
                return delivered
            assets.append(delivered)
        if options.num_samples == 1:
            return assets[0]
        return {"num_samples": options.num_samples, "assets": assets}
