# Pixal3D RunPod Serverless — Design

Date: 2026-05-13
Status: Approved for implementation

## Goal

A RunPod Serverless worker image for [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) inference, built by GitHub Actions, that reliably accepts an image and returns a GLB.

## What was wrong with the previous attempts

1. `handler.py` was never `COPY`'d into the image and the CMD launched the Gradio UI (`app.py`) instead of `runpod.serverless.start(...)`. Worker never picked up jobs → "can't connect" symptoms.
2. Launch scripts forced `ATTN_BACKEND=flash_attn_3` while only flash-attn 2 was installed → import error on first request.
3. The Dockerfile attempted to follow TRELLIS.2's `setup.sh` flow and source-compile every CUDA extension. `natten` build failed under PEP 517 isolation (`ModuleNotFoundError: torch`); workaround compiles took too long for GHA runners.
4. HF model downloads happened on every cold start without `HF_TOKEN`, with no persistent cache.

## Architecture

### Single mode: RunPod Serverless

The container's only job is `python3.10 /app/handler.py`, which calls `runpod.serverless.start(...)`. No SSH, no Gradio, no port 7860, no `app.py` invocation. The handler imports `pixal3d.pipelines` directly.

### Install path: `requirements-hfdemo.txt`, not TRELLIS.2 setup.sh

Pixal3D ships `requirements-hfdemo.txt` containing prebuilt cp310 wheels for every painful CUDA dep: `natten`, `flash_attn_3`, `cumesh`, `flex_gemm`, `o_voxel`, `nvdiffrast`, `nvdiffrec_render`. Plus `torch==2.6.0`+cu124 via `--extra-index-url`. Eliminates the TRELLIS.2 clone, the `setup.sh` chain, and all source compilation.

Base image becomes `nvidia/cuda:12.4.1-runtime-ubuntu22.04` (runtime, not devel — nothing compiles).

### Dual FlashAttention wheels + GPU autodetect

FlashAttention 3 is Hopper-only (sm_90). FlashAttention 2 runs on Ampere through Hopper. We install both wheels and let the handler choose at startup based on `torch.cuda.get_device_capability()`.

```python
if not os.environ.get("ATTN_BACKEND"):
    major, _ = torch.cuda.get_device_capability(0)
    os.environ["ATTN_BACKEND"] = "flash_attn_3" if major >= 9 else "flash_attn"
```

This env var must be set **before** importing anything from `pixal3d.modules.attention.config`, which reads it at module import. The package validates against `{xformers, flash_attn, flash_attn_3, flash_attn_4, sdpa, naive}` and defaults to `flash_attn`. Explicit `ATTN_BACKEND=` on the endpoint still overrides autodetect.

One image deploys cleanly to either an A100 endpoint or an H100 endpoint with zero env config.

### Network volume for weights and caches

All HF/torch caches root at `/runpod-volume/...`. The endpoint must have a network volume (60–80 GB) attached at that path.

| Env var | Value |
|---|---|
| `HF_HOME` | `/runpod-volume/huggingface` |
| `HUGGINGFACE_HUB_CACHE` | `/runpod-volume/huggingface/hub` |
| `TORCH_HOME` | `/runpod-volume/torch` |
| `XDG_CACHE_HOME` | `/runpod-volume/cache` |
| `TRITON_CACHE_DIR` | `/runpod-volume/triton` |
| `FLEX_GEMM_AUTOTUNE_CACHE_PATH` | `/runpod-volume/flex_gemm_autotune.json` |

### BiRefNet baked into image

`ZhengPeng7/BiRefNet` (~880 MB, ungated, permissive license) is baked into the image at build time at `/opt/hf_baked/hub`. At handler startup, if the volume cache doesn't yet have BiRefNet, the baked copy is `shutil.copytree`'d into `$HUGGINGFACE_HUB_CACHE`. This removes BiRefNet from the HF auth/download path entirely.

Other HF models (Pixal3D, MoGe, DinoV3) remain volume-cached, populated by the prefetch script.

### HF auth

Gated models (`TencentARC/Pixal3D`, `Ruicheng/moge-2-vitl`, possibly DinoV3) require `HF_TOKEN` on the endpoint. `huggingface_hub` reads it from `os.environ` automatically. Token only matters on cache miss; warm volume = no HF calls.

Handler validates HF_TOKEN presence at startup if any required HF cache entry is missing. Logs actionable error (which repo, which URL to visit) before crashing.

### Models loaded at module import

Handler does heavy load **once at process start**:
1. Set `ATTN_BACKEND` env var via autodetect
2. Seed BiRefNet cache from `/opt/hf_baked` if needed
3. `Pixal3DImageTo3DPipeline.from_pretrained("TencentARC/Pixal3D")`
4. Attach the four `DinoV3ProjFeatureExtractor` cond models (ss / shape_512 / shape_1024 / tex_1024)
5. Call `_load_naf()` on the cond models that need it (forces NAF download via `torch.hub`)
6. `MoGeModel.from_pretrained("Ruicheng/moge-2-vitl")` (preloaded, not lazy)
7. `pipeline.cuda()`
8. `runpod.serverless.start(...)`

Warm requests skip all of this.

### Per-request flow

```
job_input → {image_url | image_base64} → download/decode to tempfile
  → pipeline.preprocess_image (BiRefNet rembg, unless skip_rembg=true)
  → MoGe camera estimation
  → pipeline.run(camera_params=..., preprocess_image=False)
  → o_voxel.postprocess.to_glb
  → rotation matrix applied
  → glb.export
  → base64-encode bytes → return
```

Optional `skip_rembg=true` input flag bypasses BiRefNet for callers passing already-clean images.

### Prefetch script

`scripts/prefetch_models.py`: a standalone script run once on a one-shot CPU pod with `HF_TOKEN` set and the network volume mounted at `/runpod-volume`. Downloads all gated/ungated HF repos and the NAF torch.hub model into the volume cache. Surfaces auth/gating errors before any serverless deploy.

### Files in repo (final state)

| File | Purpose |
|---|---|
| `Dockerfile` | Build the serverless image |
| `handler.py` | Module-loading + RunPod handler |
| `scripts/prefetch_models.py` | One-shot volume-warming utility |
| `.github/workflows/build.yml` | CI build + push to GHCR |
| `.dockerignore` | Build context exclusions |
| `README.md` | Deploy instructions |

Removed: `start.sh`, `run_ui.sh`.

## Operational notes

- Image size estimate: ~14–17 GB (with BiRefNet baked).
- First cold start on a warm volume: 30s–2min.
- First cold start on a cold volume: 5–15 min (one-time, prefer prefetch).
- Per-job warm latency on A100: ~60s (estimate).
- Pinned versions track Pixal3D's `requirements-hfdemo.txt`.

## Out of scope

- Bake-Pixal3D-weights-into-image alternative — addressed in design discussion, rejected on size/legal/CI grounds.
- Training / fine-tuning paths.
- FlashAttention 4. (Listed by Pixal3D's config.py but not currently shipped.)
- Mesh-only or texture-only output modes. The handler always runs the full image-to-textured-mesh pipeline.
