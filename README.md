# Pixal3D GGUF on Runpod Serverless

A queue-based worker that turns a single image into a textured GLB. It uses [Aero-Ex Pixal3D GGUF](https://huggingface.co/Aero-Ex/Pixal3D-GGUF) Q8_0 flow checkpoints and the matching FP16 decoders with [Tencent's Pixal3D pipeline](https://github.com/TencentARC/Pixal3D). The loader restores each flow model's native PyTorch weights once at startup.

## Deploy

Import this repository in Runpod as a queue-based Serverless endpoint and use the root Dockerfile and default startup command. To update an existing GitHub-backed endpoint, deploy from its configured branch. Runpod displays build and rollout progress in **Builds** and **Releases**.

The Q8_0 files and FP16 decoder files total about 7.90 GB. With an attached Global Volume and `PIXAL3D_GLOBAL_VOLUME=1`, the worker downloads the pinned snapshot under `/runpod-volume/pixal3d-gguf-q8-v1` on first use and reuses it later. The Global Volume must already contain DINOv3 and NAF under `/runpod-volume/pixal3d-models-v1`. Runtime caches use local storage. MoGe-2 and BiRefNet ship in the image.

To prepare an empty volume ahead of time, run this inside the worker image with the volume attached.

```bash
PIXAL3D_GLOBAL_VOLUME=1 python -m scripts.prefetch_models
```

## Generate a textured GLB

```bash
curl -fsS "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"image_url":"https://example.com/object.png","target_triangles":30000,"texture_size":2048,"output_format":"url"}}'

curl -fsS "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

| Input | Value |
|---|---|
| `image_url` or `image_base64` | HTTP(S) image URL or Base64 image, up to 20 MB. |
| `resolution` | `1024` (default) or `1536`. `pipeline_type` supports the equivalent `1024_cascade` and `1536_cascade` names. |
| `target_triangles` | 1,000 to 1,000,000. Defaults to 200,000. |
| `texture_size` | 512, 1024, 2048 (default), or 4096. |
| `texture_format` | `webp` (default) or `png`. |
| `output_format` | `url`, `base64`, or `auto` (default). |
| `seed` | Defaults to 42. |
| `skip_rembg` | Boolean, defaults to false. |
| `low_vram` | Boolean, defaults to false. Offloads modules between stages when enabled. |
| `num_samples` | 1 to 4, defaults to 1. Multiple samples use URL delivery. |

Existing sampler, camera, remeshing, and UV controls still apply. Sampler overrides use `sparse_structure_sampler_params`, `shape_slat_sampler_params`, and `tex_slat_sampler_params`. Camera controls use `camera_params` or `manual_fov` and the framing options. The default UV backend is `sprint3d`.

When the task completes, read `output.glb_url` or `output.glb_base64`. The response includes the actual triangle count, texture count, model revision, and timings.

### R2 and S3 output

URL output stores the GLB in a private bucket and returns a one-hour signed URL. Inline output accepts GLB files up to 7 MB. Set these Worker environment variables for URL output.

| Variable | Value |
|---|---|
| `PIXAL3D_OUTPUT_BUCKET` | Bucket name. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Credentials for objects under `pixal3d/`. |
| `AWS_DEFAULT_REGION` | Bucket region or `auto` for Cloudflare R2. |
| `PIXAL3D_OUTPUT_ENDPOINT_URL` | S3-compatible API endpoint, if applicable. |
