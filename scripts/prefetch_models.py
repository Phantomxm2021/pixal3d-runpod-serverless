"""Populate /runpod-volume from a setup Pod before starting Serverless workers."""

import os
from pathlib import Path

from model_cache import (
    DINO_MODEL_NAME, MOGE_MODEL_NAME, NAF_CHECKPOINT, NAF_CHECKPOINT_URL,
    PIXAL3D_MODEL_NAME, configure_cache, pipeline_snapshot,
)


def main() -> None:
    configure_cache()

    import torch
    from huggingface_hub import snapshot_download

    pipeline = os.environ.get("PIXAL3D_MODEL_PATH", PIXAL3D_MODEL_NAME)
    print(f"[prefetch] Pixal3D single-image pipeline: {pipeline}", flush=True)
    pipeline_snapshot(pipeline)

    for model in (os.environ.get("MOGE_MODEL_PATH", MOGE_MODEL_NAME), DINO_MODEL_NAME):
        print(f"[prefetch] Hugging Face: {model}", flush=True)
        snapshot_download(model)

    # Loading hubconf.py needs the CUDA-specific natten wheel. A setup Pod only
    # needs to cache the checkpoint under the filename torch.hub will look for.
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / NAF_CHECKPOINT
    if not checkpoint.is_file():
        print("[prefetch] NAF checkpoint", flush=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(NAF_CHECKPOINT_URL, str(checkpoint))

    print("[prefetch] Model weights cached.", flush=True)


if __name__ == "__main__":
    main()
