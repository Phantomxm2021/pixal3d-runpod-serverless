"""Pre-warm a Pixal3D RunPod network volume.

Run this ONCE on a one-shot RunPod pod (CPU is fine) with the network volume
attached at /runpod-volume and HF_TOKEN set. It downloads every required HF
model + the NAF torch.hub model into the volume's cache paths. After that,
serverless workers boot in seconds against the warm volume.

Usage (inside the pod):

    pip install huggingface_hub torch
    export HF_TOKEN=hf_xxx
    python prefetch_models.py

Re-run only when bumping the Pixal3D model version (rare).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


HF_HOME = os.environ.setdefault("HF_HOME", "/runpod-volume/huggingface")
HF_HUB_CACHE = os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/runpod-volume/huggingface/hub")
TORCH_HOME = os.environ.setdefault("TORCH_HOME", "/runpod-volume/torch")

# Repos to pre-download. BiRefNet is also baked into the worker image, but
# pre-warming it here means a fresh volume can serve any worker even without
# the seeding step.
HF_REPOS = [
    os.environ.get("PIXAL3D_MODEL_PATH",  "TencentARC/Pixal3D"),
    os.environ.get("MOGE_MODEL_PATH",     "Ruicheng/moge-2-vitl"),
    os.environ.get("BIREFNET_MODEL_PATH", "ZhengPeng7/BiRefNet"),
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
]


def _check_token() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    sys.stderr.write(
        "HF_TOKEN is not set. The Pixal3D and MoGe repos are gated and will 401.\n"
        "  1) Visit https://huggingface.co/TencentARC/Pixal3D and accept terms.\n"
        "  2) Visit https://huggingface.co/Ruicheng/moge-2-vitl and accept terms.\n"
        "  3) Create a READ token at https://huggingface.co/settings/tokens .\n"
        "  4) export HF_TOKEN=hf_...\n"
    )
    sys.exit(1)


def _prefetch_hf() -> None:
    from huggingface_hub import snapshot_download

    for repo in HF_REPOS:
        print(f"[prefetch] hf: {repo}")
        snapshot_download(repo, cache_dir=HF_HUB_CACHE)


def _prefetch_torch_hub() -> None:
    # NAF lives on GitHub via torch.hub, not HF. No auth needed.
    print("[prefetch] torch.hub: valeoai/NAF (naf)")
    try:
        import torch
    except ImportError:
        print(
            "[prefetch] torch is not installed in this environment; skipping NAF download. "
            "The serverless worker will fetch it on first cold start instead.",
            file=sys.stderr,
        )
        return
    try:
        torch.hub.load("valeoai/NAF", "naf", pretrained=True, trust_repo=True)
    except Exception as exc:  # pragma: no cover - prefetch convenience
        print(f"[prefetch] WARN: NAF fetch failed ({exc}); workers will retry at boot.", file=sys.stderr)


def main() -> None:
    Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
    Path(TORCH_HOME).mkdir(parents=True, exist_ok=True)
    _check_token()
    _prefetch_hf()
    _prefetch_torch_hub()
    print("[prefetch] done. Volume is warm.")


if __name__ == "__main__":
    main()
