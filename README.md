# Pixal3D RunPod Serverless Worker

A RunPod **Serverless** worker image for [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) inference. Image is built by GitHub Actions and published to GHCR.

```text
GitHub Actions  →  ghcr.io/jj-ju/pixal3d-runpod-serverless:latest
                                            │
              RunPod Serverless endpoint  ──┘
                            │
                            └─ network volume mounted at /runpod-volume
```

This image is **inference only**. Not for training, not for fine-tuning, not for the Gradio UI.

## What the image contains

- `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base
- Python 3.10, PyTorch 2.6.0 + cu124
- Pixal3D's `requirements-hfdemo.txt` (prebuilt wheels for `natten`, `flash_attn_3`, `cumesh`, `flex_gemm`, `o_voxel`, `nvdiffrast`, `nvdiffrec_render`)
- FlashAttention 2 (`flash-attn==2.7.4.post1`) for Ampere/Ada GPUs
- Pixal3D's `utils3d-0.0.2` override wheel
- `ZhengPeng7/BiRefNet` baked into `/opt/hf_baked` (seeded into the volume at first boot)
- The Pixal3D repo at `/app/Pixal3D`
- `handler.py` as the CMD

## How the handler picks FA2 vs FA3

At process start the handler reads `torch.cuda.get_device_capability()`:

| GPU | sm | `ATTN_BACKEND` |
|---|---|---|
| H100 / H200 | 9.0+ | `flash_attn_3` |
| A100 / A40 / 4090 / L40S | 8.0 / 8.6 / 8.9 | `flash_attn` |

Explicit `ATTN_BACKEND=...` on the endpoint overrides autodetect.

## One-time setup

### 1. Build & push the image

GitHub → Actions → **Build Pixal3D RunPod Image** → **Run workflow**. The workflow builds and pushes:

```text
ghcr.io/jj-ju/pixal3d-runpod-serverless:latest
ghcr.io/jj-ju/pixal3d-runpod-serverless:sha-<short>
```

If GHCR is private, make the package public or configure RunPod registry credentials.

### 2. Accept access for gated HF models

Visit each repo and accept terms on your Hugging Face account:

- https://huggingface.co/TencentARC/Pixal3D
- https://huggingface.co/Ruicheng/moge-2-vitl

DinoV3 mirror (`camenduru/dinov3-vitl16-pretrain-lvd1689m`) and BiRefNet are typically ungated; verify if downloads 401.

### 3. Create a read token

https://huggingface.co/settings/tokens → New token → Read scope. Keep it handy as `HF_TOKEN`.

### 4. Create a RunPod network volume

RunPod → Storage → Network Volume. Size: **60–80 GB**. Pick a region close to your serverless endpoint.

### 5. Warm the volume

Spin up a cheap CPU pod with the volume mounted at `/runpod-volume`:

```bash
pip install huggingface_hub torch
export HF_TOKEN=hf_xxx
python /app/scripts/prefetch_models.py   # if the pod uses this image
# or copy scripts/prefetch_models.py from this repo
```

Takes ~10–20 minutes. After it finishes, tear the pod down — the volume keeps the weights.

### 6. Create the serverless endpoint

| Setting | Value |
|---|---|
| Container Image | `ghcr.io/jj-ju/pixal3d-runpod-serverless:latest` |
| Container Disk | 20 GB |
| Network Volume | the one you warmed, mounted at `/runpod-volume` |
| GPU | A100 80GB (or H100 if you want FA3 + lower latency) |
| Idle Timeout | 300+ seconds (keeps workers warm) |
| Max Workers | start at 1, scale up later |

Environment variables on the endpoint:

```text
HF_TOKEN=hf_xxx            # required for fresh volumes, harmless for warm ones
PIXAL3D_TIMEOUT=900        # optional, bump if jobs are slow
ATTN_BACKEND=              # leave unset — handler autodetects
```

That's everything. Workers boot, autodetect the GPU, set the FA backend, seed BiRefNet from the image, load all models from the volume, and start polling the job queue.

## Request format

### Image URL

```json
{
  "input": {
    "image_url": "https://example.com/input.png",
    "suffix": ".png",
    "seed": 42,
    "skip_rembg": false
  }
}
```

### Base64 image

```json
{
  "input": {
    "image_base64": "iVBORw0KGgo...",
    "suffix": ".png",
    "seed": 42
  }
}
```

Data URLs work too: `"data:image/png;base64,iVBORw0KGgo..."`.

`skip_rembg=true` skips BiRefNet background removal — use it if your image already has a clean foreground (transparent PNG or solid background).

## Response format

```json
{
  "filename": "output.glb",
  "mime_type": "model/gltf-binary",
  "glb_base64": "...",
  "size_bytes": 123456,
  "timing": {
    "fetch_seconds": 0.4,
    "inference_seconds": 58.2,
    "encode_seconds": 0.1
  }
}
```

Decode `glb_base64` to write `output.glb` on the client.

## Minimal client example

```python
import base64, requests

ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
KEY = "YOUR_RUNPOD_API_KEY"

r = requests.post(
    ENDPOINT,
    headers={"Authorization": f"Bearer {KEY}"},
    json={"input": {"image_url": "https://example.com/input.png"}},
    timeout=1200,
)
r.raise_for_status()
out = r.json().get("output") or r.json()

if "error" in out:
    raise RuntimeError(out["error"])

with open("output.glb", "wb") as f:
    f.write(base64.b64decode(out["glb_base64"]))
```

## Troubleshooting

### Worker crashes at startup with `OSError: 401`

`HF_TOKEN` isn't set or your account hasn't accepted access to a gated model. Logs will show which repo. Fix on the endpoint env config and on huggingface.co.

### Worker crashes with `flash_attn_3` import error

You forced `ATTN_BACKEND=flash_attn_3` on an Ampere GPU. Unset it (let autodetect handle it) or set `ATTN_BACKEND=flash_attn`.

### Job runs but returns `Inference finished but produced no GLB.`

Usually OOM or a CUDA assertion. Check worker logs. For OOM, try A100 80GB or H100 80GB. Increase `PIXAL3D_TIMEOUT` if hitting timeouts on cold workers.

### First request after deploy is very slow

Expected when the volume has just been warmed but the first worker on a node still does the initial FlexGEMM autotune + Triton JIT compile. Both cache to the volume; subsequent jobs are fast.

### GitHub Actions build fails on disk space

Less likely with this design (~14–17 GB image), but if it happens move the build to Depot, Docker Build Cloud, or a temporary RunPod pod with a large disk. Same Dockerfile works anywhere.

## What lives where

| Thing | Location | Persistent? |
|---|---|---|
| Docker image | GHCR, pulled to RunPod nodes | No (re-pulled per node) |
| Pixal3D source | `/app/Pixal3D` inside image | Per image |
| Pixal3D / MoGe / DinoV3 weights | `/runpod-volume/huggingface/hub` | Yes |
| BiRefNet weights | Baked at `/opt/hf_baked/hub`, seeded to volume on first boot | Yes |
| NAF model (torch.hub) | `/runpod-volume/torch/hub` | Yes |
| FlexGEMM autotune cache | `/runpod-volume/flex_gemm_autotune.json` | Yes |
| Triton JIT cache | `/runpod-volume/triton` | Yes |
| `HF_TOKEN` | RunPod endpoint env config | Per endpoint |
