"""Cache the fixed Aero-Ex Pixal3D Q8_0 release and its FP16 decoders."""

from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.global_models import _stream_file
from scripts.model_cache import RUNPOD_VOLUME, use_global_volume

REPOSITORY = "Aero-Ex/Pixal3D-GGUF"
REVISION = "1c2630e408cde95f7bac9c65028404aa49796065"
MODEL_FORMAT = "gguf-q8-0"
ARTIFACTS = {
    "Sparse/ss_flow_img_dit_1_3B_64_bf16": (
        "gguf", 503, 1_427_084_032, "6551ab0381262db6b2a6a2e351f18f7462583d2b064ca9d0b89e4410ef73f16d",
    ),
    "shape/slat_flow_img2shape_dit_1_3B_512_bf16": (
        "gguf", 535, 1_476_761_184, "651dc94f65bf8af0ddd1e6c5c266c496f1cd61c03f965acaaf53d92bc6f4912f",
    ),
    "shape/slat_flow_img2shape_dit_1_3B_1024_bf16": (
        "gguf", 535, 1_476_761_184, "7ce2303bb03fcbb8ec435d76be34dcbe21f1b3d12a673a818855342b6d5652c8",
    ),
    "texture/slat_flow_imgshape2tex_dit_1_3B_1024_bf16": (
        "gguf", 535, 1_476_813_408, "c827e6f5e4b6078434d8c2d054e37424706de1e5587830d5943121cd4ce2dcd8",
    ),
    "decoder/shape_dec_next_dc_f16c32_fp16": (
        "safetensors", 678, 948_490_494, "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581",
    ),
    "decoder/ss_dec_conv3d_16l8_fp16": (
        "safetensors", 245, 147_591_972, "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
    ),
    "decoder/tex_dec_next_dc_f16c32_fp16": (
        "safetensors", 705, 948_458_812, "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c",
    ),
}


def _artifacts() -> dict[str, tuple[str, int, str | None]]:
    files = {"pipeline.json": ("pipeline.json", 4_068, None)}
    for source, (kind, config_size, checkpoint_size, digest) in ARTIFACTS.items():
        name = source.split("/")[-1]
        files[f"ckpts/{name}.json"] = (f"{source}.json", config_size, None)
        suffix = "_Q8_0.gguf" if kind == "gguf" else ".safetensors"
        files[f"ckpts/{name}.{kind}"] = (f"{source}{suffix}", checkpoint_size, digest)
    return files


def _complete(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


def _download(root: Path, item: tuple[str, tuple[str, int, str | None]]) -> None:
    name, (source, size, digest) = item
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as file:
        temporary = Path(file.name)
    try:
        print(f"[gguf cache] Downloading {source} ({size} bytes)", flush=True)
        _stream_file(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{source}", temporary, size=size, sha256=digest)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def gguf_snapshot() -> tuple[Path, dict]:
    root = RUNPOD_VOLUME if use_global_volume() else Path("/tmp/pixal3d-cache")
    snapshot = root / "pixal3d-gguf-q8-v1" / REVISION
    missing = [item for item in _artifacts().items() if not _complete(snapshot / item[0], item[1][1])]
    if missing:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda item: _download(snapshot, item), missing))
    config = json.loads((snapshot / "pipeline.json").read_text(encoding="utf-8"))
    expected = {f"ckpts/{source.split('/')[-1]}" for source in ARTIFACTS}
    if set(config["args"]["models"].values()) != expected:
        raise ValueError("Unexpected checkpoint paths in the Aero-Ex Pixal3D pipeline.")
    return snapshot, config
