# Pixal3D RunPod Serverless Worker

A RunPod Serverless worker image for running [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) inference without building Docker locally.

This repo is designed for this flow:

```text
GitHub Actions cloud build -> GitHub Container Registry -> RunPod Serverless endpoint
```

## What this image does

- Starts from `nvidia/cuda:12.4.1-devel-ubuntu22.04`.
- Installs Python 3.10.
- Installs PyTorch `2.6.0` with CUDA 12.4 wheels.
- Installs TRELLIS.2 runtime/native dependencies.
- Installs Pixal3D inference dependencies.
- Patches Pixal3D's hardcoded `flash_attn_3` setting so the backend can be selected through `ATTN_BACKEND`.
- Runs a RunPod Serverless handler that accepts an input image and returns a base64 `.glb`.

This is for inference only, not training or fine-tuning.

## Build the image without local Docker

1. Open this repository on GitHub.
2. Go to **Actions**.
3. Select **Build Pixal3D RunPod Image**.
4. Click **Run workflow**.

The workflow builds and pushes:

```text
ghcr.io/jj-ju/pixal3d-runpod-serverless:latest
```

If the GHCR package is private, either make it public or configure RunPod registry credentials.

## Important build note

This image is heavy. It compiles native CUDA/PyTorch extensions and may exceed the disk/time limits of GitHub-hosted Actions runners. If the GitHub Actions build fails due to disk space, memory, or timeout, use the same Dockerfile with one of these cloud build options instead:

- Docker Build Cloud
- Depot
- A temporary GPU or CPU RunPod pod with a large volume
- Another remote builder that supports large Docker builds

You still do **not** need to build the image on your local machine.

## RunPod Serverless endpoint config

Suggested endpoint settings:

```yaml
Container Image: ghcr.io/jj-ju/pixal3d-runpod-serverless:latest
Container Disk: 80 GB minimum, 120 GB safer
Timeout: 900 seconds or higher
GPU: A100 40GB minimum recommended; A100 80GB or H100 preferred
```

Environment variables:

```text
ATTN_BACKEND=flash_attn
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PIXAL3D_TIMEOUT=900
HF_HOME=/runpod-volume/huggingface
HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub
TORCH_HOME=/runpod-volume/torch
XDG_CACHE_HOME=/runpod-volume/cache
```

The `/runpod-volume` cache paths are useful if you attach a RunPod network volume. Without a network volume, you can remove or ignore them.

## Request format

### Image URL

```json
{
  "input": {
    "image_url": "https://example.com/input.png",
    "suffix": ".png"
  }
}
```

### Base64 image

```json
{
  "input": {
    "image_base64": "iVBORw0KGgo...",
    "suffix": ".png"
  }
}
```

Data URLs are also accepted:

```json
{
  "input": {
    "image_base64": "data:image/png;base64,iVBORw0KGgo...",
    "suffix": ".png"
  }
}
```

## Response format

```json
{
  "filename": "output.glb",
  "mime_type": "model/gltf-binary",
  "glb_base64": "...",
  "size_bytes": 123456,
  "logs_tail": "..."
}
```

Decode `glb_base64` client-side to write `output.glb`.

## Minimal Python client decode example

```python
import base64
import requests

RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
RUNPOD_API_KEY = "YOUR_RUNPOD_API_KEY"

payload = {
    "input": {
        "image_url": "https://example.com/input.png",
        "suffix": ".png",
    }
}

response = requests.post(
    RUNPOD_ENDPOINT,
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    json=payload,
    timeout=1200,
)
response.raise_for_status()
result = response.json()

# Depending on RunPod sync/async response shape, your output may be nested under output.
output = result.get("output", result)

if "error" in output:
    raise RuntimeError(output["error"])

with open("output.glb", "wb") as f:
    f.write(base64.b64decode(output["glb_base64"]))
```

## H100 / FlashAttention 3 note

The image defaults to:

```text
ATTN_BACKEND=flash_attn
```

That is the safer default for A100/4090-style environments using FlashAttention 2. If you build a Hopper/H100-specific image with FlashAttention 3 installed, you can set:

```text
ATTN_BACKEND=flash_attn_3
```

## Troubleshooting

### GitHub Actions fails with no space left on device

Use a bigger remote builder. This is expected to be a large build because TRELLIS.2 and Pixal3D include native/CUDA dependencies.

### RunPod worker times out

Increase:

```text
PIXAL3D_TIMEOUT=1800
```

Also remember first run may download model weights unless you have warmed or cached them.

### Import error for attention backend

Keep:

```text
ATTN_BACKEND=flash_attn
```

Do not set `flash_attn_3` unless the image actually includes FlashAttention 3.

### Out of memory

Use A100 80GB or H100 first. After you have a working endpoint, you can experiment with smaller GPUs.
