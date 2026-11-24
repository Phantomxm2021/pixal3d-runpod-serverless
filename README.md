# Pixal3D Runpod Serverless Worker

A queue-based Runpod Serverless worker for [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D). Send an image URL or a base64-encoded image and receive a textured GLB.

## Deploy from GitHub

1. In Runpod **Settings → Connections**, connect GitHub and grant Runpod access to this repository or your fork.
2. In **Serverless → New Endpoint**, choose **Import Git Repository**. Select the branch with the fix, keep **Dockerfile Path** at `Dockerfile` in the repository root, and select the **Queue** endpoint type. The root-level `rp_handler.py` registers the queue handler and is the container startup command.
3. Start with one **A100 80 GB** or **H100 80 GB** worker and an **1800-second execution timeout**. Leave the Docker command at its default. Set `HF_TOKEN` as an endpoint secret if Hugging Face requires access to any selected model. Monitor progress in the endpoint's **Builds** tab.
4. Attach a network volume in the same data center to reuse downloaded weights between workers. Serverless mounts it at `/runpod-volume`. The worker can populate an empty volume when it first starts, or you can warm the volume before deployment using the commands below.

Runpod can display `runpod.serverless.start() handler not found in your repo` when importing from a branch other than `main`. If Runpod shows the expected commit, finds the root `Dockerfile`, and the selected branch contains `rp_handler.py`, click **Next → Continue anyway** to reach endpoint configuration. Existing GitHub-backed endpoints use GitHub Releases to trigger updates. [Runpod GitHub deployment documentation](https://docs.runpod.io/serverless/workers/github-integration)

### Warm the network volume

On a setup Pod with the same volume mounted at `/runpod-volume`, run these commands from a checkout of this repository. Python 3.10+ and a CPU installation of PyTorch are sufficient. Set `HF_TOKEN` on the Pod if your models require it.

```bash
python -m pip install 'huggingface_hub<1' torch
python scripts/prefetch_models.py
```

The script downloads the Pixal3D single-image checkpoints, MoGe-2, DINOv3, and the NAF checkpoint. The worker uses the BiRefNet model baked into the image for background removal. If you override `PIXAL3D_MODEL_PATH` or `MOGE_MODEL_PATH`, use the same settings on the setup Pod and the endpoint.

### Deploy from a container image

The GitHub Actions workflow publishes `ghcr.io/<owner>/pixal3d-runpod-serverless:latest` and `ghcr.io/<owner>/pixal3d-runpod-serverless:sha-<full-commit-sha>`. It runs on pushes to `main` or via **Run workflow** on the desired branch. Leave `pixal3d_ref` blank to use the pinned Pixal3D source version, or enter an upstream branch or full commit SHA.

Select **Docker** in Runpod and enter the image tag. For a private GHCR package, configure container registry credentials. The immutable commit tag makes deployments and rollbacks explicit.

## Submit a job

Use `/run` for asynchronous jobs. Query `/status/<job-id>` until the job reaches a terminal state. On `COMPLETED`, read the GLB from `output`. For jobs that fit the synchronous wait window, send the same payload to `/runsync`. [Runpod queue API reference](https://docs.runpod.io/serverless/endpoints/operation-reference)

```bash
curl -fsS "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"image_url":"https://example.com/object.png","seed":42}}'

curl -fsS "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

| `input` field | Value |
|---|---|
| `image_url` | HTTP(S) URL to an image. Takes precedence when both image fields are set. |
| `image_base64` | Raw base64 or a `data:image/...;base64,...` URL. |
| `suffix` | Optional input extension. Defaults to `.png`. |
| `seed` | Integer. Defaults to `42`. |
| `skip_rembg` | Boolean. Defaults to `false`. Set `true` for images that are already prepared as clean RGB inputs. |
| `output_format` | `base64` (default) or `url`. |

The response includes `filename`, `mime_type`, `size_bytes`, `timing`, and either `glb_base64` or `glb_url`. Invalid inputs and inference errors surface as a Runpod job error. Inference exception details are available in the worker logs.

### Return large GLB files through S3

Inline output supports GLBs up to 7 MB so the Base64 response fits both Runpod queue operations. For larger GLBs, set `output_format` to `url` and configure these environment variables on the endpoint:

| Variable | Value |
|---|---|
| `PIXAL3D_OUTPUT_BUCKET` | S3 bucket name. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Credentials with permission to write and read objects under `pixal3d/`. |
| `AWS_DEFAULT_REGION` | Bucket region. |
| `PIXAL3D_OUTPUT_ENDPOINT_URL` | Optional service URL for an S3-compatible provider. |

The worker returns a presigned `glb_url` that expires after one hour. Use a bucket lifecycle rule if the stored GLBs should expire as well.

## Test locally

```bash
python -m unittest discover -s tests
```

Run a GPU smoke test with a built image and a mounted model volume:

```bash
docker run --rm --gpus all -e HF_TOKEN \
  -v /path/to/model-volume:/runpod-volume "$IMAGE" \
  python3 -u rp_handler.py --test_input '{"input":{"image_url":"https://example.com/object.png"}}'
```

Replace the sample URL with a reachable image. On a new volume the first start includes model downloads, and the first job also compiles GPU kernels. A GPU error mentioning `flash_attn_3` on an Ampere or Ada card can indicate an `ATTN_BACKEND` override. Unset the override to use automatic selection (`flash_attn_3` on Hopper, `flash_attn` on Ampere and Ada).
