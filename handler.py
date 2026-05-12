import base64
import os
import subprocess
import tempfile
import traceback
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import runpod


PIXAL_DIR = Path("/app/Pixal3D")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("PIXAL3D_TIMEOUT", "900"))


def _download_image(url: str, out_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "RunPod-Pixal3D-Worker"})
    with urlopen(request, timeout=60) as response:
        out_path.write_bytes(response.read())


def _decode_base64_image(data: str, out_path: Path) -> None:
    # Accept both raw base64 and data URLs like:
    # data:image/png;base64,iVBORw0KGgo...
    if data.strip().startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    out_path.write_bytes(base64.b64decode(data))


def _run_pixal3d(input_image: Path, output_glb: Path) -> str:
    env = os.environ.copy()
    env.setdefault("ATTN_BACKEND", "flash_attn")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        "python3.10",
        "inference.py",
        "--image",
        str(input_image),
        "--output",
        str(output_glb),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PIXAL_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stdout[-8000:])

    if not output_glb.exists() or output_glb.stat().st_size == 0:
        raise RuntimeError("Pixal3D finished but did not create a non-empty GLB file.")

    return result.stdout[-4000:]


def handler(job):
    """RunPod Serverless handler.

    Input formats:
      {"input": {"image_url": "https://.../input.png"}}
      {"input": {"image_base64": "...", "suffix": ".png"}}

    Output:
      {
        "filename": "output.glb",
        "mime_type": "model/gltf-binary",
        "glb_base64": "...",
        "size_bytes": 123,
        "logs_tail": "..."
      }
    """
    try:
        job_input = job.get("input") or {}
        image_url = job_input.get("image_url")
        image_base64 = job_input.get("image_base64")

        if not image_url and not image_base64:
            return {"error": "Provide input.image_url or input.image_base64."}

        suffix = job_input.get("suffix", ".png")
        if not suffix.startswith("."):
            suffix = f".{suffix}"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            input_image = tmp_dir / f"input_{uuid.uuid4().hex}{suffix}"
            output_glb = tmp_dir / f"output_{uuid.uuid4().hex}.glb"

            if image_url:
                _download_image(image_url, input_image)
            else:
                _decode_base64_image(image_base64, input_image)

            logs_tail = _run_pixal3d(input_image, output_glb)
            glb_bytes = output_glb.read_bytes()

            return {
                "filename": "output.glb",
                "mime_type": "model/gltf-binary",
                "glb_base64": base64.b64encode(glb_bytes).decode("utf-8"),
                "size_bytes": len(glb_bytes),
                "logs_tail": logs_tail,
            }

    except subprocess.TimeoutExpired:
        return {
            "error": f"Pixal3D timed out after {DEFAULT_TIMEOUT_SECONDS} seconds.",
            "hint": "Increase PIXAL3D_TIMEOUT on the RunPod endpoint if the GPU is slow or cold-start downloads are happening.",
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "traceback": traceback.format_exc()[-4000:],
        }


runpod.serverless.start({"handler": handler})
