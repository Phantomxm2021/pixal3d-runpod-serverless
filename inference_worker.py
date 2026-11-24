"""Load the Pixal3D models once and export textured meshes on the worker GPU."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

from scripts.model_cache import (
    DEFAULT_BIREFNET_MODEL,
    DINO_MODEL_NAME,
    GLOBAL_MODEL_ROOT,
    MOGE_MODEL_NAME,
    NAF_CHECKPOINT,
    PIXAL3D_MODEL_NAME,
    configure_cache,
    pipeline_snapshot,
    use_global_volume,
)

PIXAL_DIR = Path("/app/Pixal3D")


def _log(message: str) -> None:
    print(f"[inference] {message}", flush=True)


_log("Configuring writable caches and importing PyTorch")
configure_cache()
import torch


def _autodetect_attn_backend() -> str:
    """Pick FA2 vs FA3 based on GPU compute capability.

    Honors an explicit ATTN_BACKEND if already set (e.g. on the endpoint).
    """
    explicit = os.environ.get("ATTN_BACKEND")
    if explicit:
        _log(f"ATTN_BACKEND explicit={explicit}")
        return explicit

    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    backend = "flash_attn_3" if major == 9 else "flash_attn"
    _log(f"GPU detected: {name} sm_{major}{minor} -> ATTN_BACKEND={backend}")

    os.environ["ATTN_BACKEND"] = backend
    return backend


_log("Checking CUDA availability")
if not torch.cuda.is_available():
    raise RuntimeError("Pixal3D inference requires a CUDA-enabled GPU.")
_autodetect_attn_backend()
sys.path.insert(0, str(PIXAL_DIR))

_log("Importing o_voxel, FlexGEMM and model classes")
import numpy as np
import o_voxel
from PIL import Image
from moge.model.v2 import MoGeModel
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
)

def _prepare_global_models() -> None:
    from scripts.global_models import verify_global_models

    verify_global_models(GLOBAL_MODEL_ROOT)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / NAF_CHECKPOINT
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.symlink_to(GLOBAL_MODEL_ROOT / NAF_CHECKPOINT)


GLOBAL_MODELS = use_global_volume()
if GLOBAL_MODELS:
    _log("Verifying prefetched Global Volume model files")
    _prepare_global_models()
    _log("Global Volume model files verified")


IMAGE_COND_CONFIGS = {
    "ss":          {"model_name": DINO_MODEL_NAME, "image_size":  512, "grid_resolution": 16},
    "shape_512":   {"model_name": DINO_MODEL_NAME, "image_size":  512, "grid_resolution": 32, "use_naf_upsample": True, "naf_target_size":  512},
    "shape_1024":  {"model_name": DINO_MODEL_NAME, "image_size": 1024, "grid_resolution": 64, "use_naf_upsample": True, "naf_target_size":  512},
    "tex_1024":    {"model_name": DINO_MODEL_NAME, "image_size": 1024, "grid_resolution": 64, "use_naf_upsample": True, "naf_target_size": 1024},
}


def _build_pipeline() -> Pixal3DImageTo3DPipeline:
    model_path = os.environ.get("PIXAL3D_MODEL_PATH", PIXAL3D_MODEL_NAME)
    if GLOBAL_MODELS:
        model_path = str(GLOBAL_MODEL_ROOT / "pixal3d")
    _log(f"loading Pixal3D pipeline from {model_path}")
    snapshot, config = pipeline_snapshot(model_path)
    rembg_path = os.environ.get("BIREFNET_MODEL_PATH", DEFAULT_BIREFNET_MODEL)
    baked_path = Path("/opt/birefnet")
    if rembg_path == DEFAULT_BIREFNET_MODEL and (baked_path / "config.json").is_file():
        rembg_path = str(baked_path)
    config["args"]["rembg_model"]["args"]["model_name"] = rembg_path

    # Keep the downloaded snapshot immutable and give Pixal3D local paths to its weights.
    with tempfile.TemporaryDirectory(prefix="pixal3d-pipeline-") as workdir:
        pipeline_dir = Path(workdir)
        for entry in snapshot.iterdir():
            if entry.name != "pipeline.json":
                (pipeline_dir / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        (pipeline_dir / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(str(pipeline_dir))

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
    if GLOBAL_MODELS:
        cfg = {**cfg, "model_name": str(GLOBAL_MODEL_ROOT / "dinov3")}
    m = DinoV3ProjFeatureExtractor(**cfg)
    m.eval()
    return m


def _build_moge() -> "MoGeModel":
    name = os.environ.get("MOGE_MODEL_PATH", MOGE_MODEL_NAME)
    if GLOBAL_MODELS:
        name = str(GLOBAL_MODEL_ROOT / "moge/model.pt")
    _log(f"loading MoGe-2 from {name}")
    m = MoGeModel.from_pretrained(name).to("cuda")
    m.eval()
    return m


_load_start = time.monotonic()
PIPELINE = _build_pipeline()
MOGE_MODEL = _build_moge()
_log(f"model load complete in {time.monotonic() - _load_start:.1f}s")


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


def _estimate_camera(image: Image.Image, mesh_scale: float, extend_pixel: int, image_resolution: int) -> dict:
    width = image.width
    img_t = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).cuda()
    with torch.no_grad():
        out = MOGE_MODEL.infer(img_t)
    intrinsics = out["intrinsics"].squeeze().cpu().numpy()
    fx = intrinsics[0, 0] * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))
    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    target = torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel])
    distance = _distance_from_fov(camera_angle_x, grid_point, target, mesh_scale, image_resolution)
    return {"camera_angle_x": camera_angle_x, "distance": float(distance), "mesh_scale": mesh_scale}


def run_inference(input_image: Path, output_glb: Path, *, seed: int, skip_rembg: bool) -> dict:
    t_start = time.monotonic()
    _log("Preprocessing input image")
    with Image.open(input_image) as image:
        if skip_rembg:
            image_for_pipeline = image.convert("RGB")
            image_for_pipeline.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        else:
            image_for_pipeline = PIPELINE.preprocess_image(image)

    _log("Estimating camera with MoGe-2")
    camera_params = _estimate_camera(image_for_pipeline, mesh_scale=1.0, extend_pixel=0, image_resolution=512)

    _log("Running Pixal3D mesh and texture generation")
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

    _log("Exporting textured GLB")
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

    glb.export(str(output_glb), extension_webp=True)
    return {"inference_seconds": round(time.monotonic() - t_start, 2)}
