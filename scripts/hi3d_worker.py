"""Run the pinned Hi3D pipeline inside its own Python environment."""

from __future__ import annotations

import argparse
import json
import logging
import math
import resource
import time
from pathlib import Path


def fit_output_frame(mesh, source) -> None:
    """Restore the source GLB's center and longest dimension after Hi3D normalization."""
    target_extent = float(mesh.extents.max())
    source_extent = float(source.extents.max())
    if not all(math.isfinite(size) and size > 0 for size in (target_extent, source_extent)):
        raise ValueError("Cannot align a mesh with a non-finite or empty extent.")
    mesh.apply_scale(source_extent / target_extent)
    mesh.apply_translation(source.bounds.mean(axis=0) - mesh.bounds.mean(axis=0))


def refine(args: argparse.Namespace) -> dict:
    import torch
    import trimesh
    from PIL import Image, ImageOps
    from sparc3d import Sparc3DPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Hi3D requires a CUDA-enabled GPU.")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    print(f"[hi3d] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    timing = {}
    started = time.monotonic()

    def finish_stage(stage: str) -> None:
        nonlocal started
        elapsed = time.monotonic() - started
        timing[f"{stage}_seconds"] = round(elapsed, 2)
        print(f"[hi3d] {stage} completed in {elapsed:.2f}s", flush=True)
        started = time.monotonic()

    def progress(event) -> None:
        if event.step == 1 or event.step % 5 == 0 or event.step == event.num_inference_steps:
            print(f"[hi3d] Pass {event.pass_index + 1}/{event.num_passes} "
                  f"step {event.step}/{event.num_inference_steps}", flush=True)

    with Image.open(args.image) as original:
        image = ImageOps.exif_transpose(original).convert("RGBA")
    if not args.skip_rembg and image.getchannel("A").getextrema() == (255, 255):
        from rembg import new_session
        from sparc3d.utils.image import remove_background

        session = new_session("u2net", providers=["CPUExecutionProvider"])
        image = remove_background(image, session=session)
        del session
    mesh = trimesh.load(args.source, force="mesh", process=False)
    finish_stage("hi3d_preprocess")

    print("[hi3d] Loading model and image encoders on CPU", flush=True)
    pipeline = Sparc3DPipeline.from_pretrained(
        args.model_root / "hi3d", local_files_only=True,
        dino_model_id=args.model_root / "dinov2", clip_model_id=args.model_root / "clip",
        progress_callback=progress,
    )
    if args.resolution == 1024:
        _ = pipeline.transformer_1024
    finish_stage("hi3d_cpu_load")

    pipeline.to("cuda")
    torch.cuda.synchronize()
    finish_stage("hi3d_cuda_transfer")

    print(f"[hi3d] Starting {args.resolution} refinement", flush=True)
    result = pipeline(image, mesh, resolution=args.resolution, num_inference_steps=args.steps, seed=args.seed)
    torch.cuda.synchronize()
    finish_stage("hi3d_refinement")

    refined = result.mesh
    if args.target_triangles is not None and len(refined.faces) > args.target_triangles:
        from sparc3d.sparcubes.postprocess import postprocess_mesh

        refined = postprocess_mesh(refined, target_face_count=args.target_triangles, component_face_ratio=0.0)
    fit_output_frame(refined, mesh)
    refined.export(args.output)
    finish_stage("hi3d_glb_export")

    return {
        "mesh_resolution": result.resolution, "num_inference_steps": args.steps, "timing": timing,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resolution", type=int, choices=(512, 1024), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--target-triangles", type=int)
    parser.add_argument("--skip-rembg", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.report.write_text(json.dumps(refine(args)), encoding="utf-8")


if __name__ == "__main__":
    main()
