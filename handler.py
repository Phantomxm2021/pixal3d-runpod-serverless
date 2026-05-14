"""Pixal3D RunPod Serverless handler.

Module-level model loading: imports, autodetects FA backend, seeds BiRefNet
cache, then loads Pixal3D + MoGe once. Every job after that runs against
GPU-resident models.
"""

from __future__ import annotations

import base64
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


PIXAL_DIR = Path("/app/Pixal3D")
TIMEOUT_SECONDS = int(os.environ.get("PIXAL3D_TIMEOUT", "900"))


def _log(msg: str) -> None:
    print(f"[handler] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Pre-import setup: env vars, volume dirs, BiRefNet cache seed, FA autodetect.
# Everything below MUST run before importing anything from `pixal3d` because
# `pixal3d.modules.attention.config` reads ATTN_BACKEND at module import.
# ---------------------------------------------------------------------------

def _ensure_cache_dirs() -> None:
    paths = [
        os.environ.get("HF_HOME", "/runpod-volume/huggingface"),
        os.environ.get("HUGGINGFACE_HUB_CACHE", "/runpod-volume/huggingface/hub"),
        os.environ.get("TORCH_HOME", "/runpod-volume/torch"),
        os.environ.get("XDG_CACHE_HOME", "/runpod-volume/cache"),
        os.environ.get("TRITON_CACHE_DIR", "/runpod-volume/triton"),
    ]
    for p in paths:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _log(f"WARN: could not create cache dir {p}: {exc}")


def _seed_baked_models() -> None:
    """Copy image-baked HF snapshots into the volume cache if missing.

    Today this is just BiRefNet, baked at /opt/hf_baked/hub during the Docker
    build. Cheap to extend to other small ungated models later.
    """
    baked_root = Path(os.environ.get("PIXAL3D_BAKED_HF_CACHE", "/opt/hf_baked/hub"))
    dst_root = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/runpod-volume/huggingface/hub"))
    if not baked_root.is_dir():
        _log(f"no baked HF cache at {baked_root}; skipping seed")
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    for entry in baked_root.iterdir():
        if not entry.name.startswith("models--"):
            continue
        target = dst_root / entry.name
        if target.exists():
            continue
        _log(f"seeding {entry.name} from baked cache -> {target}")
        try:
            shutil.copytree(entry, target, symlinks=True)
        except OSError as exc:
            _log(f"WARN: failed to seed {entry.name}: {exc}")


def _autodetect_attn_backend() -> str:
    """Pick FA2 vs FA3 based on GPU compute capability.

    Honors an explicit ATTN_BACKEND if already set (e.g. on the endpoint).
    """
    explicit = os.environ.get("ATTN_BACKEND")
    if explicit:
        _log(f"ATTN_BACKEND explicit={explicit}")
        return explicit

    try:
        import torch  # local import: cheap, no pixal3d touched yet
    except Exception as exc:
        _log(f"WARN: torch import failed during autodetect: {exc}; defaulting to flash_attn")
        os.environ["ATTN_BACKEND"] = "flash_attn"
        return "flash_attn"

    if not torch.cuda.is_available():
        _log("WARN: CUDA not available at autodetect time; defaulting to flash_attn")
        backend = "flash_attn"
    else:
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        backend = "flash_attn_3" if major >= 9 else "flash_attn"
        _log(f"GPU detected: {name} sm_{major}{minor} -> ATTN_BACKEND={backend}")

    os.environ["ATTN_BACKEND"] = backend
    return backend


def _warn_if_hf_token_missing() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/runpod-volume/huggingface/hub"))
    pixal_cached = (cache / f"models--{os.environ.get('PIXAL3D_MODEL_PATH', 'TencentARC/Pixal3D').replace('/', '--')}").is_dir()
    if pixal_cached:
        _log("HF_TOKEN not set, but Pixal3D appears cached on volume; proceeding")
        return
    _log(
        "HF_TOKEN not set AND the volume cache is empty. "
        "Set HF_TOKEN on the RunPod endpoint and ensure your HF account has accepted access for: "
        f"{os.environ.get('PIXAL3D_MODEL_PATH', 'TencentARC/Pixal3D')}, "
        f"{os.environ.get('MOGE_MODEL_PATH', 'Ruicheng/moge-2-vitl')}. "
        "Cold start will fail with 401."
    )


_ensure_cache_dirs()
_seed_baked_models()
_autodetect_attn_backend()
_warn_if_hf_token_missing()

# Pixal3D source isn't pip-installed; add it to sys.path before importing.
sys.path.insert(0, str(PIXAL_DIR))

# ---------------------------------------------------------------------------
# Heavy imports + one-time model load.
# ---------------------------------------------------------------------------

_load_start = time.time()
_log("importing torch + pixal3d...")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import runpod  # noqa: E402

from pixal3d.pipelines import Pixal3DImageTo3DPipeline  # noqa: E402
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (  # noqa: E402
    DinoV3ProjFeatureExtractor,
)
import o_voxel  # noqa: E402

# MoGe is installed via `git+https://github.com/microsoft/MoGe.git` from
# Pixal3D's requirements-hfdemo.txt. Import lazily so the user gets a clear
# message if it's missing.
try:
    from moge.model.v2 import MoGeModel
except Exception as exc:  # pragma: no cover - import-time diagnostic
    _log(f"FATAL: MoGe import failed: {exc}")
    raise

DINO_MODEL_NAME = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
IMAGE_COND_CONFIGS = {
    "ss":          {"model_name": DINO_MODEL_NAME, "image_size":  512, "grid_resolution": 16},
    "shape_512":   {"model_name": DINO_MODEL_NAME, "image_size":  512, "grid_resolution": 32, "use_naf_upsample": True, "naf_target_size":  512},
    "shape_1024":  {"model_name": DINO_MODEL_NAME, "image_size": 1024, "grid_resolution": 64, "use_naf_upsample": True, "naf_target_size":  512},
    "tex_1024":    {"model_name": DINO_MODEL_NAME, "image_size": 1024, "grid_resolution": 64, "use_naf_upsample": True, "naf_target_size": 1024},
}


def _build_pipeline() -> Pixal3DImageTo3DPipeline:
    model_path = os.environ.get("PIXAL3D_MODEL_PATH", "TencentARC/Pixal3D")
    _log(f"loading Pixal3D pipeline from {model_path}")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    _log("building DinoV3 cond models")
    pipeline.image_cond_model_ss          = _build_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512   = _build_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024  = _build_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024    = _build_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    pipeline.low_vram = False
    pipeline.cuda()
    for attr in ("image_cond_model_ss", "image_cond_model_shape_512",
                 "image_cond_model_shape_1024", "image_cond_model_tex_1024"):
        getattr(pipeline, attr).cuda()

    _log("forcing NAF download for cond models that need it")
    for attr in ("image_cond_model_ss", "image_cond_model_shape_512",
                 "image_cond_model_shape_1024", "image_cond_model_tex_1024"):
        m = getattr(pipeline, attr)
        if getattr(m, "use_naf_upsample", False):
            m._load_naf()

    return pipeline


def _build_cond_model(cfg: dict) -> DinoV3ProjFeatureExtractor:
    m = DinoV3ProjFeatureExtractor(**cfg)
    m.eval()
    return m


def _build_moge() -> "MoGeModel":
    name = os.environ.get("MOGE_MODEL_PATH", "Ruicheng/moge-2-vitl")
    _log(f"loading MoGe-2 from {name}")
    m = MoGeModel.from_pretrained(name).to("cuda")
    m.eval()
    return m


PIPELINE = _build_pipeline()
MOGE_MODEL = _build_moge()
_log(f"model load complete in {time.time() - _load_start:.1f}s")


# ---------------------------------------------------------------------------
# Camera estimation (lifted from Pixal3D's inference.py, unchanged math).
# ---------------------------------------------------------------------------

def _compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    return float((focal_length * resolution / 32.0).item())


def _distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, _ = gp[0].item(), gp[1].item(), gp[2].item()
    xt, _yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = _compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    return f_pixels * xw / x_ndc - yw


def _estimate_camera(image_path: Path, mesh_scale: float, extend_pixel: int, image_resolution: int) -> dict:
    pil = Image.open(image_path).convert("RGB")
    width, _ = pil.size
    img_t = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).permute(2, 0, 1).cuda()
    with torch.no_grad():
        out = MOGE_MODEL.infer(img_t)
    intrinsics = out["intrinsics"].squeeze().cpu().numpy()
    fx = intrinsics[0, 0] * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))
    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    target = torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel])
    distance = _distance_from_fov(camera_angle_x, grid_point, target, mesh_scale, image_resolution)
    return {"camera_angle_x": camera_angle_x, "distance": float(distance), "mesh_scale": mesh_scale}


# ---------------------------------------------------------------------------
# Job-level: image fetch, inference, GLB export.
# ---------------------------------------------------------------------------

def _download_image(url: str, out: Path) -> None:
    req = Request(url, headers={"User-Agent": "RunPod-Pixal3D-Worker"})
    with urlopen(req, timeout=60) as resp:
        out.write_bytes(resp.read())


def _decode_image(data: str, out: Path) -> None:
    if data.strip().startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    out.write_bytes(base64.b64decode(data))


def _run_inference(input_image: Path, output_glb: Path, *, seed: int, skip_rembg: bool) -> dict:
    t_start = time.time()
    img = Image.open(input_image)

    if skip_rembg:
        image_for_pipeline = img.convert("RGB")
    else:
        image_for_pipeline = PIPELINE.preprocess_image(img)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        image_for_pipeline.save(tmp_path)
        camera_params = _estimate_camera(tmp_path, mesh_scale=1.0, extend_pixel=0, image_resolution=512)
    finally:
        tmp_path.unlink(missing_ok=True)

    torch.manual_seed(seed)
    mesh_list, (_shape_slat, _tex_slat, grid_res) = PIPELINE.run(
        image_for_pipeline,
        camera_params=camera_params,
        seed=seed,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=49152,
    )
    mesh = mesh_list[0]

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=PIPELINE.pbr_attr_layout,
        grid_size=grid_res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=200000, texture_size=2048,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False,
    )

    rot = np.array([
        [-1,  0,  0, 0],
        [ 0,  0, -1, 0],
        [ 0, -1,  0, 0],
        [ 0,  0,  0, 1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(output_glb), extension_webp=True)
    return {"inference_seconds": round(time.time() - t_start, 2)}


def handler(job):
    """RunPod Serverless entrypoint.

    Input:
      {"input": {"image_url": "https://..."}}
      {"input": {"image_base64": "..."}}
      Optional: "suffix" (default ".png"), "seed" (default 42),
                "skip_rembg" (default false).

    Output:
      {"filename": "output.glb", "mime_type": "model/gltf-binary",
       "glb_base64": "...", "size_bytes": N, "timing": {...}}
    """
    try:
        ji = job.get("input") or {}
        image_url = ji.get("image_url")
        image_b64 = ji.get("image_base64")
        if not image_url and not image_b64:
            return {"error": "Provide input.image_url or input.image_base64."}

        suffix = ji.get("suffix", ".png")
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        seed = int(ji.get("seed", 42))
        skip_rembg = bool(ji.get("skip_rembg", False))

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            in_path = tmpdir / f"input_{uuid.uuid4().hex}{suffix}"
            out_path = tmpdir / f"output_{uuid.uuid4().hex}.glb"

            t0 = time.time()
            if image_url:
                _download_image(image_url, in_path)
            else:
                _decode_image(image_b64, in_path)
            t_fetch = time.time() - t0

            timing = _run_inference(in_path, out_path, seed=seed, skip_rembg=skip_rembg)

            if not out_path.exists() or out_path.stat().st_size == 0:
                return {"error": "Inference finished but produced no GLB."}

            t0 = time.time()
            glb_bytes = out_path.read_bytes()
            encoded = base64.b64encode(glb_bytes).decode("utf-8")
            t_encode = time.time() - t0

            return {
                "filename": "output.glb",
                "mime_type": "model/gltf-binary",
                "glb_base64": encoded,
                "size_bytes": len(glb_bytes),
                "timing": {
                    "fetch_seconds": round(t_fetch, 2),
                    "encode_seconds": round(t_encode, 2),
                    **timing,
                },
            }

    except Exception as exc:
        return {
            "error": str(exc),
            "traceback": traceback.format_exc()[-4000:],
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
