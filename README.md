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

### Use a Global Volume

Attach the Global Volume at `/runpod-volume` and set `PIXAL3D_GLOBAL_VOLUME=1`. The worker reads regular model files under `/runpod-volume/pixal3d-models-v1` and keeps its writable Hugging Face, Torch, and Triton caches under `/tmp/pixal3d-cache`.

To populate a new volume, temporarily set `PIXAL3D_PREFILL_ONLY=1` and **Max workers** to `1`. Submit `{"input":{"action":"prefill_models"},"policy":{"executionTimeout":1800000,"ttl":1800000}}` to `/run` and wait for `output.status` to become `ready`. This streams only the files used by the single-image pipeline and publishes a manifest after all files are complete. Remove `PIXAL3D_PREFILL_ONLY` before submitting an inference job. The first inference loads the prefetched weights from the volume.

### Load Pixal3D from the Runpod node cache

Set the endpoint's **Model** to `TencentARC/Pixal3D` and add `PIXAL3D_RUNPOD_MODEL_CACHE=1`. The worker loads the single-image Pixal3D checkpoints from Runpod's cached Hugging Face snapshot under `/runpod-volume/huggingface-cache/hub` and fails if the required files are missing. Keep `PIXAL3D_GLOBAL_VOLUME=1` with the prefetched volume attached for DINOv3 and NAF. The default MoGe-2 checkpoint and BiRefNet are included in the container image. [Runpod cached models](https://docs.runpod.io/serverless/endpoints/model-caching)

Set `PIXAL3D_MODEL_REVISION` to the full Hugging Face commit SHA shown in the endpoint's **Model** field when that field is pinned. The worker verifies `pipeline.json` and its checkpoints against the file sizes for that commit. Send `{"input":{"action":"prepare_node_cache"}}` to verify and repair a writable node cache without running inference. A `ready` response includes the verified revision, file count, and total bytes.

Compare the job's `delayTime` in milliseconds with `output.timing.model_load_seconds` on the first job that loads the models. The load time includes the Pixal3D pipeline, auxiliary models, and transfer to the GPU. `pixal3d_pipeline_load_seconds` measures the initial pipeline load before that transfer.

The conditioning stages share one DINOv3 instance and one NAF instance. With a Global Volume, the worker stages DINOv3 on local scratch and loads directly in BF16 on GPUs with native BF16 support. Projection and NAF inputs remain FP32. Each request reuses DINOv3 features for stages with the same image resolution. A cold job reports worker import time, Pixal3D snapshot resolution, checkpoint read, model construction, state-dict load and CUDA transfer, and the MoGe-2 loading stages. Per-checkpoint times appear in the worker log. All generation jobs report preprocessing, camera estimation, sampling, shape upsampling and quantization, decoding, post-processing, encoding, and delivery time.

During MoGe-2 startup, the container logs process RSS and its high-water mark along with available cgroup usage, peak, limit, and OOM counters. An observer reports the inference process exit status and counter changes if it survives the exit.

### Deploy from a container image

The GitHub Actions workflow publishes `ghcr.io/<owner>/pixal3d-runpod-serverless:latest` and `ghcr.io/<owner>/pixal3d-runpod-serverless:sha-<full-commit-sha>`. It runs on pushes to `main` or via **Run workflow** on the desired branch. Leave `pixal3d_ref` blank to use the pinned Pixal3D source version, or enter an upstream branch or full commit SHA.

Select **Docker** in Runpod and enter the image tag. For a private GHCR package, configure container registry credentials. The immutable commit tag makes deployments and rollbacks explicit.

The Docker build compiles NATTEN 0.21.0 against PyTorch 2.6 and CUDA 12.4 and verifies that it contains SM80 kernels for Ampere and Ada GPUs (including A40, RTX 4090, and L40S). Hopper workers select the separately installed upstream SM90 wheel. The compiler is confined to the build stage.

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
| `output_format` | `auto` (default), `base64`, or `url`. |
| `asset_format` | `glb` (default) or `usdz`. USDZ requires an output bucket and uses URL delivery. |

The response includes `filename`, `mime_type`, `size_bytes`, the exported mesh's `triangle_count` and `vertex_count`, `timing`, and either `glb_base64` or `glb_url`. Invalid inputs and inference errors surface as a Runpod job error. Inference exception details are available in the worker logs.

### Return large GLB files through S3

Inline output supports GLBs up to 7 MB so the Base64 response fits both Runpod queue operations. The default `auto` uploads larger files when a bucket is configured. Request `url` to upload every GLB or `base64` to require inline delivery. Configure these environment variables on the endpoint for URL output.

| Variable | Value |
|---|---|
| `PIXAL3D_OUTPUT_BUCKET` | S3 bucket name. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Credentials with permission to write and read objects under `pixal3d/`. |
| `AWS_DEFAULT_REGION` | Bucket region. Use `auto` for Cloudflare R2. |
| `PIXAL3D_OUTPUT_ENDPOINT_URL` | Optional S3-compatible API endpoint, such as the R2 S3 API URL shown in the bucket settings. |

The worker returns a presigned `glb_url` that expires after one hour. Use a bucket lifecycle rule if the stored GLBs should expire as well.

### Generate an editable source and export GLB variants

Use the existing GPU endpoint with the same image fields and request `asset_format: "usdz"`. The worker retains the decoder's native quad mesh, bakes base color and metallic-roughness textures, and stores a self-contained USDZ in the private output bucket. The response includes `asset_id`, `usdz_url`, `quad_count` and `vertex_count`. Keep the asset ID to request new GLB variants after the download URL expires. The USDZ stores the full source topology and GLB decimation runs on the CPU.

Submit the following payload to the same endpoint's `/run` operation. Blender is bundled in the worker image and runs its export in background mode on the CPU. An export uses the GPU endpoint's worker time and skips Pixal3D inference.

```json
{
  "input": {
    "action": "export_glb",
    "asset_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "target_triangles": 30000,
    "texture_size": 2048
  }
}
```

Both export options are optional. The default is a 30,000-triangle target and 2048px maximum texture dimension. Blender accepts targets from 1,000 to 200,000 triangles and texture sizes of 512, 1024 and 2048. The response includes the actual triangle and vertex counts, a `glb_url`, and `cached`. Repeating the same export parameters reuses the stored GLB. GLB uses triangles and the USDZ retains the editable quads.

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
