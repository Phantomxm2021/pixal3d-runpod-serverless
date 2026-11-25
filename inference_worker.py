"""Load the Pixal3D models once and export textured meshes on the worker GPU."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from scripts.generation_options import GenerationOptions
from scripts.global_models import stage_dinov3
from scripts.model_cache import (
    DEFAULT_BIREFNET_MODEL,
    DINO_MODEL_NAME,
    GLOBAL_MODEL_ROOT,
    MOGE_MODEL_NAME,
    NAF_CHECKPOINT,
    PIXAL3D_MODEL_NAME,
    configure_cache,
    pipeline_snapshot,
    runpod_pipeline_snapshot,
    use_global_volume,
    use_runpod_model_cache,
)
from scripts.worker_diagnostics import memory_stage

PIXAL_DIR = Path("/app/Pixal3D")
HOPPER_NATTEN_DIR = Path("/opt/natten-hopper")
LOCAL_MOGE_CHECKPOINT = Path("/opt/moge/model.pt")


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
if torch.cuda.get_device_capability(0)[0] == 9 and HOPPER_NATTEN_DIR.is_dir():
    sys.path.insert(0, str(HOPPER_NATTEN_DIR))
    _log("Using the Hopper NATTEN wheel")
sys.path.insert(0, str(PIXAL_DIR))

_log("Importing o_voxel, FlexGEMM and model classes")
import numpy as np
import o_voxel
from PIL import Image
from transformers import DINOv3ViTModel
from moge.model.v2 import MoGeModel
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
    cache_dinov3_features,
)


def _prepare_global_models() -> None:
    from scripts.global_models import verify_global_models

    verify_global_models(GLOBAL_MODEL_ROOT, include_pipeline=not RUNPOD_MODEL_CACHE)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / NAF_CHECKPOINT
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.symlink_to(GLOBAL_MODEL_ROOT / NAF_CHECKPOINT)


GLOBAL_MODELS = use_global_volume()
RUNPOD_MODEL_CACHE = use_runpod_model_cache()
if GLOBAL_MODELS:
    _log("Verifying prefetched Global Volume model files")
    _prepare_global_models()
    _log("Global Volume model files verified")


IMAGE_COND_CONFIGS = {
    "ss":          {"image_size":  512, "grid_resolution": 16},
    "shape_512":   {"image_size":  512, "grid_resolution": 32, "use_naf_upsample": True, "naf_target_size":  512},
    "shape_1024":  {"image_size": 1024, "grid_resolution": 64, "use_naf_upsample": True, "naf_target_size":  512},
    "tex_1024":    {"image_size": 1024, "grid_resolution": 64, "use_naf_upsample": True, "naf_target_size": 1024},
}


def _load_pixal3d_pipeline() -> tuple[Pixal3DImageTo3DPipeline, dict[str, float]]:
    started = time.monotonic()
    if RUNPOD_MODEL_CACHE:
        _log("Resolving Pixal3D weights in the Runpod node cache")
        snapshot, config = runpod_pipeline_snapshot()
    else:
        model_path = os.environ.get("PIXAL3D_MODEL_PATH", PIXAL3D_MODEL_NAME)
        if GLOBAL_MODELS:
            model_path = str(GLOBAL_MODEL_ROOT / "pixal3d")
        snapshot, config = pipeline_snapshot(model_path)
    snapshot_seconds = time.monotonic() - started
    _log(f"Pixal3D snapshot resolved in {snapshot_seconds:.2f}s")
    _log(f"Loading Pixal3D pipeline from {snapshot}")
    rembg_path = os.environ.get("BIREFNET_MODEL_PATH", DEFAULT_BIREFNET_MODEL)
    baked_path = Path("/opt/birefnet")
    if rembg_path == DEFAULT_BIREFNET_MODEL and (baked_path / "config.json").is_file():
        rembg_path = str(baked_path)
    config["args"]["rembg_model"]["args"]["model_name"] = rembg_path

    stage_seconds = {"construct": 0.0, "checkpoint_read": 0.0, "state_dict": 0.0}

    def record_stage(model: str, stage: str, seconds: float) -> None:
        stage_seconds[stage] += seconds
        _log(f"Pixal3D {model} {stage} complete in {seconds:.2f}s")

    # Keep the downloaded snapshot immutable and give Pixal3D local paths to its weights.
    with tempfile.TemporaryDirectory(prefix="pixal3d-pipeline-") as workdir:
        pipeline_dir = Path(workdir)
        for entry in snapshot.iterdir():
            if entry.name != "pipeline.json":
                (pipeline_dir / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        (pipeline_dir / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(str(pipeline_dir), load_timing=record_stage)
    return pipeline, {
        "pixal3d_snapshot_seconds": round(snapshot_seconds, 2),
        **{f"pixal3d_{stage}_seconds": round(seconds, 2) for stage, seconds in stage_seconds.items()},
    }


def _prepare_pipeline(pipeline: Pixal3DImageTo3DPipeline, *, low_vram: bool) -> dict[str, float]:
    dino_path = str(GLOBAL_MODEL_ROOT / "dinov3") if GLOBAL_MODELS else DINO_MODEL_NAME
    pipeline.low_vram = low_vram
    _log("Selecting CUDA for Pixal3D with stage offload" if low_vram else "Transferring Pixal3D pipeline to CUDA")
    started = time.monotonic()
    pipeline.cuda()
    torch.cuda.synchronize()
    pipeline_cuda_seconds = round(time.monotonic() - started, 2)
    _log(f"Pixal3D pipeline CUDA transfer complete in {pipeline_cuda_seconds:.2f}s")

    timings = {"pixal3d_pipeline_cuda_seconds": pipeline_cuda_seconds}
    with ExitStack() as stack:
        load_path = dino_path
        if GLOBAL_MODELS:
            _log("Copying DINOv3 weights from Global Volume to local scratch")
            started = time.monotonic()
            try:
                load_path = str(stack.enter_context(
                    stage_dinov3(Path(dino_path), Path(os.environ["XDG_CACHE_HOME"]))
                ))
            except OSError:
                _log("Local DINOv3 staging failed, loading from Global Volume")
            else:
                timings["dinov3_local_stage_seconds"] = round(time.monotonic() - started, 2)
                _log(f"DINOv3 local staging complete in {timings['dinov3_local_stage_seconds']:.2f}s")

        dino_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float32
        _log(f"Loading shared DINOv3 from {load_path} with dtype={dino_dtype}")
        started = time.monotonic()
        dino_model = DINOv3ViTModel.from_pretrained(load_path, dtype=dino_dtype)
        timings["dinov3_load_seconds"] = round(time.monotonic() - started, 2)
        _log(f"DINOv3 CPU load complete in {timings['dinov3_load_seconds']:.2f}s")

        started = time.monotonic()
        if not low_vram:
            _log("Transferring shared DINOv3 to CUDA")
            dino_model.cuda()
            torch.cuda.synchronize()
        timings["dinov3_cuda_seconds"] = round(time.monotonic() - started, 2)
        _log(f"DINOv3 CUDA transfer complete in {timings['dinov3_cuda_seconds']:.2f}s")

    started = time.monotonic()
    cond_models = []
    for stage, config in IMAGE_COND_CONFIGS.items():
        cond_model = DinoV3ProjFeatureExtractor(dino_path, dino_model=dino_model, **config)
        cond_model.eval()
        cond_models.append(cond_model)
        setattr(pipeline, f"image_cond_model_{stage}", cond_model)
    projection_load_seconds = time.monotonic() - started
    _log(f"Projection modules initialized in {projection_load_seconds:.2f}s")

    started = time.monotonic()
    if not low_vram:
        for cond_model in cond_models:
            cond_model.proj_grid.cuda()
        torch.cuda.synchronize()
    projection_cuda_seconds = time.monotonic() - started
    if not low_vram:
        _log(f"Projection modules moved to CUDA in {projection_cuda_seconds:.2f}s")

    timings.update({
        "dinov3_and_projection_load_seconds": round(timings["dinov3_load_seconds"] + projection_load_seconds, 2),
        "dinov3_and_projection_cuda_seconds": round(timings["dinov3_cuda_seconds"] + projection_cuda_seconds, 2),
    })
    naf_users = [cond_model for cond_model in cond_models if cond_model.use_naf_upsample]
    if naf_users:
        _log("Loading shared NAF for image conditioning")
        started = time.monotonic()
        first_user, *other_users = naf_users
        first_user._load_naf()
        for cond_model in other_users:
            cond_model.naf_model = first_user.naf_model
        torch.cuda.synchronize()
        timings["naf_load_seconds"] = round(time.monotonic() - started, 2)
        _log(f"Shared NAF ready for {len(naf_users)} projection modules in {timings['naf_load_seconds']:.2f}s")
    return timings


def _read_moge_checkpoint() -> tuple[dict, dict[str, float]]:
    name = os.environ.get("MOGE_MODEL_PATH", MOGE_MODEL_NAME)
    path = Path(name)
    timings = {}
    with ExitStack() as stack:
        if name == MOGE_MODEL_NAME and LOCAL_MOGE_CHECKPOINT.is_file():
            path = LOCAL_MOGE_CHECKPOINT
        elif name == MOGE_MODEL_NAME and GLOBAL_MODELS:
            _log("Staging the Global Volume MoGe-2 checkpoint to local scratch")
            memory_stage("moge_local_stage")
            started = time.monotonic()
            directory = stack.enter_context(tempfile.TemporaryDirectory(
                prefix="moge-", dir=os.environ["XDG_CACHE_HOME"],
            ))
            path = Path(directory) / "model.pt"
            shutil.copyfile(GLOBAL_MODEL_ROOT / "moge/model.pt", path)
            timings["moge_local_stage_seconds"] = round(time.monotonic() - started, 2)
            _log(f"MoGe-2 local staging complete in {timings['moge_local_stage_seconds']:.2f}s")
        elif not path.is_file():
            from huggingface_hub import hf_hub_download

            _log(f"Resolving MoGe-2 checkpoint for {name}")
            path = Path(hf_hub_download(repo_id=name, repo_type="model", filename="model.pt"))

        _log(f"Reading MoGe-2 checkpoint on CPU from {path}")
        memory_stage("moge_checkpoint_read")
        started = time.monotonic()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        read_seconds = round(time.monotonic() - started, 2)
        _log(f"MoGe-2 checkpoint read complete in {read_seconds:.2f}s")
        return checkpoint, {**timings, "moge_checkpoint_read_seconds": read_seconds}


def _build_moge(*, low_vram: bool) -> tuple[MoGeModel, dict[str, float]]:
    checkpoint, read_timings = _read_moge_checkpoint()

    # Match the pinned MoGeModel.from_pretrained constructor and state-dict options.
    _log("Creating MoGe-2 model on CPU")
    memory_stage("moge_create_model")
    init_started = time.monotonic()
    m = MoGeModel(**checkpoint["model_config"])
    create_seconds = round(time.monotonic() - init_started, 2)
    _log(f"MoGe-2 model creation complete in {create_seconds:.2f}s")

    _log("Loading MoGe-2 state_dict on CPU")
    memory_stage("moge_load_state_dict")
    started = time.monotonic()
    m.load_state_dict(checkpoint["model"], strict=False)
    state_dict_seconds = round(time.monotonic() - started, 2)
    _log(f"MoGe-2 state_dict load complete in {state_dict_seconds:.2f}s")
    del checkpoint
    cpu_init_seconds = round(time.monotonic() - init_started, 2)
    _log(f"MoGe-2 CPU initialization complete in {cpu_init_seconds:.2f}s")

    started = time.monotonic()
    if not low_vram:
        _log("Transferring MoGe-2 to CUDA")
        memory_stage("moge_cuda_transfer")
        m = m.to("cuda")
        torch.cuda.synchronize()
    cuda_seconds = round(time.monotonic() - started, 2)
    _log(f"MoGe-2 CUDA transfer complete in {cuda_seconds:.2f}s")
    m.eval()
    return m, {
        **read_timings,
        "moge_cpu_init_seconds": cpu_init_seconds,
        "moge_model_create_seconds": create_seconds,
        "moge_state_dict_load_seconds": state_dict_seconds,
        "moge_cuda_transfer_seconds": cuda_seconds,
    }


PIPELINE = None
MOGE_MODEL = None
MODEL_LOAD_TIMINGS = {}


def prepare_models(*, low_vram: bool = False) -> bool:
    global PIPELINE, MOGE_MODEL
    if PIPELINE is not None:
        return False
    started = time.monotonic()
    try:
        pipeline, pipeline_timings = _load_pixal3d_pipeline()
        pipeline_seconds = time.monotonic() - started
        _log(f"Pixal3D pipeline load complete in {pipeline_seconds:.2f}s")
        conditioning_timings = _prepare_pipeline(pipeline, low_vram=low_vram)
        moge, moge_timings = _build_moge(low_vram=low_vram)
    except Exception:
        _log(f"Model load failed after {time.monotonic() - started:.2f}s")
        raise
    finally:
        memory_stage("")
    PIPELINE, MOGE_MODEL = pipeline, moge
    MODEL_LOAD_TIMINGS.update({
        "pixal3d_pipeline_load_seconds": round(pipeline_seconds, 2),
        "model_load_seconds": round(time.monotonic() - started, 2),
        **pipeline_timings, **conditioning_timings, **moge_timings,
    })
    _log(f"Model load complete in {MODEL_LOAD_TIMINGS['model_load_seconds']:.2f}s")
    return True


def release_models() -> None:
    """Free Pixal3D weights before an isolated refinement process uses the GPU."""
    global PIPELINE, MOGE_MODEL
    if PIPELINE is not None:
        PIPELINE = MOGE_MODEL = None
        MODEL_LOAD_TIMINGS.clear()
        gc.collect()
        torch.cuda.empty_cache()
        _log("Released Pixal3D models for Hi3D")


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


def _estimate_fov(image: Image.Image) -> float:
    width = image.width
    img_t = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).cuda()
    with torch.no_grad():
        out = MOGE_MODEL.infer(img_t)
    intrinsics = out["intrinsics"].squeeze().cpu().numpy()
    fx = intrinsics[0, 0] * width
    return 2 * math.atan(width / (2 * fx))


def _camera_for_image(
    image: Image.Image, *, camera_params: dict | None = None, manual_fov: float | None = None,
    mesh_scale: float = 1.0, extend_pixel: int = 0, image_resolution: int = 512,
) -> dict:
    if camera_params is not None:
        return camera_params
    camera_angle_x = manual_fov
    if camera_angle_x is None:
        if PIPELINE.low_vram:
            MOGE_MODEL.cuda()
        try:
            camera_angle_x = _estimate_fov(image)
        finally:
            if PIPELINE.low_vram:
                MOGE_MODEL.cpu()
    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    target = torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel])
    distance = _distance_from_fov(camera_angle_x, grid_point, target, mesh_scale, image_resolution)
    return {"camera_angle_x": camera_angle_x, "distance": float(distance), "mesh_scale": mesh_scale}


def _set_low_vram(enabled: bool, *, force: bool = False) -> None:
    if not force and PIPELINE.low_vram == enabled:
        return
    conditioning = [getattr(PIPELINE, f"image_cond_model_{stage}") for stage in IMAGE_COND_CONFIGS]
    if enabled:
        for model in PIPELINE.models.values():
            model.cpu()
            if hasattr(model, "low_vram"):
                model.low_vram = False
        PIPELINE.rembg_model.cpu()
        for model in conditioning:
            model.cpu()
        MOGE_MODEL.cpu()
        PIPELINE.low_vram = True
        # In low-VRAM mode, this selects CUDA without moving model weights.
        PIPELINE.cuda()
    else:
        PIPELINE.low_vram = False
        try:
            PIPELINE.cuda()
            for model in conditioning:
                model.cuda()
            MOGE_MODEL.cuda()
        except Exception:
            _set_low_vram(True, force=True)
            raise


def run_job_sample(input_image: Path, output_path: Path, *, seed: int, skip_rembg: bool, options: GenerationOptions):
    try:
        return run_inference(input_image, output_path, seed=seed, skip_rembg=skip_rembg, options=options)
    except Exception:
        if options.low_vram:
            # A failure can interrupt a stage before it moves its model back to CPU.
            _set_low_vram(True, force=True)
        raise


GLB_ROTATION = np.array([
    [-1,  0,  0, 0],
    [ 0,  0, -1, 0],
    [ 0, -1,  0, 0],
    [ 0,  0,  0, 1],
], dtype=np.float64)


def run_inference(
    input_image: Path, output_path: Path, *, seed: int, skip_rembg: bool,
    options: GenerationOptions,
) -> tuple[dict[str, float], dict[str, int]]:
    t_start = time.monotonic()
    stage_start = t_start
    timings = {}

    def finish_stage(name: str) -> None:
        nonlocal stage_start
        torch.cuda.synchronize()
        now = time.monotonic()
        seconds = round(now - stage_start, 2)
        timings[f"{name}_seconds"] = seconds
        _log(f"{name} complete in {seconds:.2f}s")
        stage_start = time.monotonic()

    _set_low_vram(options.low_vram)
    _log("Preprocessing input image")
    with Image.open(input_image) as image:
        if skip_rembg:
            image_for_pipeline = image.convert("RGB")
            image_for_pipeline.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        else:
            image_for_pipeline = PIPELINE.preprocess_image(image)
    finish_stage("preprocess")

    _log("Resolving camera parameters")
    camera_params = _camera_for_image(image_for_pipeline, **options.camera)
    finish_stage("camera")

    _log("Running Pixal3D white mesh generation" if options.white_mesh else "Running Pixal3D mesh and texture generation")
    generation_start = time.monotonic()
    as_usdz = output_path.suffix == ".usdz"
    torch.manual_seed(seed)
    with cache_dinov3_features(image_for_pipeline):
        mesh_list, (_shape_slat, _tex_slat, grid_res) = PIPELINE.run(
            image_for_pipeline,
            camera_params=camera_params,
            seed=seed,
            preprocess_image=False,
            return_latent=True,
            stage_finished=finish_stage,
            retain_quads=as_usdz,
            geometry_only=options.white_mesh,
            **options.pipeline,
        )
    timings["generation_seconds"] = round(time.monotonic() - generation_start, 2)
    if "shape_decoder_upsample_seconds" in timings:
        timings["shape_upsample_seconds"] = round(
            timings["shape_decoder_upsample_seconds"] + timings["shape_quantize_seconds"], 2
        )
    mesh = mesh_list[0]

    if options.white_mesh:
        import trimesh

        target = options.postprocess["decimation_target"]
        if len(mesh.faces) > target:
            mesh.simplify(target)
        finish_stage("white_mesh_simplify")
        if not len(mesh.faces):
            raise RuntimeError("Pixal3D decoded an empty white mesh.")
        glb = trimesh.Trimesh(vertices=mesh.vertices.cpu().numpy(), faces=mesh.faces.cpu().numpy(), process=False)
        glb.apply_transform(GLB_ROTATION)
        glb.export(str(output_path))
        finish_stage("glb_encode")
        return {"inference_seconds": round(time.monotonic() - t_start, 2), **timings}, {
            "mesh_resolution": grid_res, "triangle_count": len(glb.faces), "vertex_count": len(glb.vertices),
        }

    quad_asset = None
    if as_usdz:
        from scripts.quad_asset import QuadAsset

        quad_asset = QuadAsset(mesh.vertices, mesh.faces, mesh.quad_faces)

    if as_usdz:
        _log("Preparing textured USDZ")
    else:
        _log(f"Preparing GLB with {options.postprocess['decimation_target']} target triangles "
             f"and {options.postprocess['texture_size']}px textures")
    _log(f"UV backend={options.postprocess['uv_backend']}, decoded triangles={len(mesh.faces)}")
    prefix = "usdz" if as_usdz else "glb"
    postprocess_started = time.monotonic()

    def finish_postprocess_stage(name: str) -> None:
        finish_stage(f"{prefix}_{name}")

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=PIPELINE.pbr_attr_layout,
        grid_size=grid_res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        quad_asset=quad_asset,
        stage_finished=finish_postprocess_stage,
        **options.postprocess,
    )
    finish_postprocess_stage("material")
    postprocess_seconds = round(time.monotonic() - postprocess_started, 2)
    timings[f"{prefix}_postprocess_seconds"] = postprocess_seconds
    _log(f"{prefix}_postprocess complete in {postprocess_seconds:.2f}s")

    glb.apply_transform(GLB_ROTATION)

    if quad_asset is not None:
        geometry = quad_asset.export(glb, output_path)
        finish_stage("usdz_encode")
    else:
        glb.export(str(output_path), extension_webp=options.texture_format == "webp")
        finish_stage("glb_encode")
        geometry = {"triangle_count": len(glb.faces), "vertex_count": len(glb.vertices)}
    return {"inference_seconds": round(time.monotonic() - t_start, 2), **timings}, {
        "mesh_resolution": grid_res, **geometry,
    }
