# Pixal3D RunPod Serverless Worker

A RunPod **Serverless** worker image for [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) inference. You send an image, the worker returns a `.glb`. Image is built in GitHub Actions and published to GHCR; nothing is built locally.

```text
GitHub Actions  →  ghcr.io/<owner>/pixal3d-runpod-serverless:latest
                                    │
              RunPod Serverless endpoint
                                    │
                    network volume at /runpod-volume
                       (Pixal3D + MoGe + DinoV3 + NAF weights)
```

---

## Requirements

You must have all of these set up **before** the worker can answer a single job.

### 1. Hugging Face account with access to the gated models

| Repo | Why | Action |
|---|---|---|
| [`TencentARC/Pixal3D`](https://huggingface.co/TencentARC/Pixal3D) | Main pipeline weights | Visit the page, click "Agree and access repository" |
| [`Ruicheng/moge-2-vitl`](https://huggingface.co/Ruicheng/moge-2-vitl) | MoGe-2 camera estimator | Visit, accept terms |
| [`camenduru/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m) | DinoV3 image features (4×) | Usually ungated; accept terms if prompted |
| [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) | Background removal (baked into image) | Usually ungated |

If your account isn't approved for a gated repo, downloads will 401 even with a valid token.

### 2. A Hugging Face **read** token

Generate one at https://huggingface.co/settings/tokens → **New token** → **Read** scope. Store it; you'll paste it into the RunPod endpoint config as `HF_TOKEN`.

### 3. A RunPod account

With:
- Billing set up so you can run a CPU pod (for ~$0.30 of warming) and a GPU serverless endpoint.
- Permission to create network volumes and serverless endpoints.

### 4. A GitHub account with this repo

Forked or cloned. GitHub Actions runs the image build under `${{ github.actor }}` and pushes to `ghcr.io/<your-username>/pixal3d-runpod-serverless`.

### 5. (Optional) Public GHCR package or RunPod registry credentials

By default GHCR packages created via the workflow are **private**. RunPod can't pull a private image without credentials. Either:
- Make the GHCR package public (Package Settings → Change visibility → Public), or
- Configure registry credentials on the RunPod endpoint (Settings → Container Registry Credentials).

### 6. A GPU class that the worker supports

The image autodetects FA2 vs FA3 at boot:

| GPU | Compute capability | Backend chosen |
|---|---|---|
| A100 (40GB / 80GB), A40 | 8.0 | `flash_attn` |
| RTX 4090, L40, L40S | 8.9 | `flash_attn` |
| H100, H200 | 9.0+ | `flash_attn_3` |

Pixal3D's reference deployment runs on H100. A100 80GB is the cost-optimal choice on serverless and is recommended.

---

## Instructions

Do these once, in order. Total time: ~30–45 minutes (most of it waiting on the volume warmup).

### Step 1: Build the image

1. Push this repo (or your fork) to GitHub.
2. Go to **Actions** → **Build Pixal3D RunPod Image** → **Run workflow**. Leave `pixal3d_ref` as `master` unless you want to pin a specific commit.
3. Wait ~10–15 minutes. The workflow publishes two tags:
   ```text
   ghcr.io/<owner>/pixal3d-runpod-serverless:latest
   ghcr.io/<owner>/pixal3d-runpod-serverless:sha-<short>
   ```
4. If you want, make the package public: GitHub → your profile → **Packages** → click the package → **Package settings** → **Change visibility**.

### Step 2: Create the network volume

1. RunPod → **Storage** → **Network Volume** → **+ New Network Volume**.
2. Size: **60 GB** (80 GB if you want headroom).
3. Region: pick one with A100 (or your target GPU) availability. **Your serverless endpoint must live in the same region.**
4. Name it something memorable, e.g. `pixal3d-weights`.

### Step 3: Warm the volume

The volume starts empty. Workers can't download ~15 GB on every cold start, so we pre-fill it once.

1. RunPod → **Pods** → **Deploy** → pick a small CPU instance (e.g. 4 vCPU / 16 GB RAM). GPU not needed.
2. Attach the network volume from Step 2 at mount path `/runpod-volume`.
3. Use any image with Python 3.10+ — RunPod's default Ubuntu image is fine. Or use this repo's image; the prefetch script is at `/app/scripts/prefetch_models.py`.
4. Add environment variable `HF_TOKEN=hf_xxx` (your token from Requirement 2).
5. Deploy and connect via web terminal.
6. Run:
   ```bash
   # If your pod uses this image:
   python /app/scripts/prefetch_models.py

   # If your pod uses a generic Python image:
   pip install huggingface_hub torch
   curl -O https://raw.githubusercontent.com/<owner>/pixal3d-runpod-serverless/main/scripts/prefetch_models.py
   export HF_TOKEN=hf_xxx
   python prefetch_models.py
   ```
7. Wait ~10–20 minutes. The script prints each repo as it downloads.
8. When you see `[prefetch] done. Volume is warm.`, **terminate the pod**. The volume keeps the weights.

You only need to re-run prefetch when Tencent ships a new Pixal3D model version — rare, maybe 2–3× a year.

### Step 4: Create the serverless endpoint

1. RunPod → **Serverless** → **+ New Endpoint**.
2. Endpoint configuration:

   | Field | Value |
   |---|---|
   | Endpoint name | `pixal3d` (or whatever) |
   | Container image | `ghcr.io/<owner>/pixal3d-runpod-serverless:latest` |
   | Container disk | 20 GB |
   | Network volume | the one from Step 2, mount path `/runpod-volume` |
   | GPU type | A100 80GB (recommended) or H100 80GB |
   | Max workers | 1 to start (raise after testing) |
   | Idle timeout | 300 seconds |
   | Execution timeout | 1800 seconds |
   | Region | same as the network volume |

3. Environment variables:

   ```text
   HF_TOKEN=hf_xxx           # required (belt-and-suspenders even with warm volume)
   PIXAL3D_TIMEOUT=900       # per-job timeout in seconds; optional
   ```

   Do **not** set `ATTN_BACKEND` — let the handler autodetect it. Only override for debugging (e.g., `sdpa` to bypass flash-attn entirely).

4. Deploy. The endpoint takes a minute to provision.

### Step 5: Send a test job

```bash
curl -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer <your-runpod-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/some-object.png",
      "seed": 42
    }
  }'
```

Or in Python:

```python
import base64, requests

ENDPOINT = "https://api.runpod.ai/v2/<endpoint-id>/runsync"
KEY = "<your-runpod-api-key>"

r = requests.post(
    ENDPOINT,
    headers={"Authorization": f"Bearer {KEY}"},
    json={"input": {"image_url": "https://example.com/object.png"}},
    timeout=1200,
)
r.raise_for_status()
out = r.json().get("output") or r.json()

if "error" in out:
    raise RuntimeError(out["error"])

with open("output.glb", "wb") as f:
    f.write(base64.b64decode(out["glb_base64"]))
print("size:", out["size_bytes"], "timing:", out["timing"])
```

**The very first request after deploying** is slow (1–3 minutes) — the worker reads weights from the volume, runs FlexGEMM autotune, and JIT-compiles Triton kernels. The autotune and Triton caches are written back to the volume, so subsequent cold starts are fast (~30–60 s) and warm-worker requests are ~30–60 s of pure inference.

---

## API reference

### Request

```json
{
  "input": {
    "image_url":    "https://example.com/input.png",   // OR image_base64
    "image_base64": "iVBORw0KGgo...",                  // OR data:image/png;base64,...
    "suffix":       ".png",                            // optional, default ".png"
    "seed":         42,                                // optional, default 42
    "skip_rembg":   false                              // optional, skip BiRefNet bg removal
  }
}
```

Set `skip_rembg: true` if your image already has a clean foreground (transparent PNG or solid background). Saves ~1–2 s and removes BiRefNet as a single point of failure for your request.

### Response — success

```json
{
  "filename":    "output.glb",
  "mime_type":   "model/gltf-binary",
  "glb_base64":  "Z2xURg...",
  "size_bytes":  4823104,
  "timing": {
    "fetch_seconds":     0.4,
    "inference_seconds": 58.2,
    "encode_seconds":    0.1
  }
}
```

### Response — error

```json
{
  "error": "...",
  "traceback": "..."
}
```

---

## Troubleshooting

### Worker crashes at startup with `OSError: 401 Client Error`

`HF_TOKEN` isn't set, OR your account hasn't accepted access for one of the gated repos. Worker logs name the repo. Fix:
- Set `HF_TOKEN` on the endpoint env config.
- Visit each gated repo on HF and accept terms.
- Redeploy.

### Worker crashes with `flash_attn_3` import or kernel error on A100

Something forced `ATTN_BACKEND=flash_attn_3` on a non-Hopper GPU. Unset the endpoint env var and let autodetect handle it. FA3 only runs on H100/H200.

### Worker times out on first request

First request after a cold volume can take 5+ minutes (FlexGEMM autotune + Triton JIT compile). Set `Execution timeout` on the endpoint to 1800+ seconds and `PIXAL3D_TIMEOUT=1800`. Subsequent requests are fast.

### Volume runs out of space

60 GB should be sufficient. If you hit limits, check the volume's usage from a temporary pod and clear `/runpod-volume/triton` or `/runpod-volume/cache`.

### Job returns `Inference finished but produced no GLB.`

Usually OOM during 1024-resolution texture generation. Move to A100 80GB or H100. Lower-VRAM cards (40 GB) may not be enough at default settings.

### GitHub Actions build fails on disk space

Less likely with this design (~14–17 GB image), but if it does:
- Re-run the workflow once or twice (transient).
- Or switch to a larger builder: Depot, Docker Build Cloud, or a temporary RunPod CPU pod with a large disk. Same Dockerfile works.

### Build succeeds, `latest` updates, but workers still pull old image

GHCR caches aggressively. Use the `sha-<short>` tag from the workflow instead of `latest` — guaranteed unique per build.

---

## What lives where

| Thing | Location | Persists? |
|---|---|---|
| Docker image | GHCR, pulled per RunPod node | Per-node cache |
| Pixal3D source | `/app/Pixal3D` inside image | Per image |
| Pixal3D / MoGe / DinoV3 weights | `/runpod-volume/huggingface/hub` | Yes |
| BiRefNet weights | Baked at `/opt/hf_baked/hub`, seeded to volume on first boot | Yes |
| NAF model (torch.hub) | `/runpod-volume/torch/hub` | Yes |
| FlexGEMM autotune cache | `/runpod-volume/flex_gemm_autotune.json` | Yes |
| Triton JIT cache | `/runpod-volume/triton` | Yes |
| `HF_TOKEN` | RunPod endpoint env config | Per endpoint |

---

## Bumping Pixal3D versions

Pixal3D ships occasional model updates. When you want to pick one up:

1. Re-run the workflow with `pixal3d_ref` set to the new commit SHA or branch.
2. Re-run the prefetch script on a CPU pod against the same volume (it'll only download new/changed snapshots).
3. Update the endpoint to the new `sha-<short>` tag.

You can keep the old image tag deployed on a separate endpoint while you A/B test.

---

## Limitations

- Inference only. No training or fine-tuning paths.
- One image config per endpoint. If you want both A100 and H100 endpoints, deploy two endpoints pointing at the same image (or different volumes).
- Output is base64 inline. GLBs over ~20 MB may exceed RunPod's `runsync` response size limit — use `/run` (async) for large outputs.
- The `spaces` library used by Pixal3D's `app.py` is bypassed entirely. This image does not serve a Gradio UI.
