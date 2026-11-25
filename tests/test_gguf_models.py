import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import gguf_models


class GGUFCacheTests(unittest.TestCase):
    def test_release_contains_only_q8_models_and_fp16_decoders(self):
        files = gguf_models._artifacts()
        checkpoints = [(source, size, digest) for source, size, digest in files.values() if digest]
        self.assertEqual(len(checkpoints), 7)
        self.assertEqual(sum(source.endswith("_Q8_0.gguf") for source, _, _ in checkpoints), 4)
        self.assertEqual(sum(source.endswith(".safetensors") for source, _, _ in checkpoints), 3)
        self.assertTrue(all(size > 0 and len(digest) == 64 for _, size, digest in checkpoints))

    def test_cache_reuses_a_complete_snapshot_and_repairs_only_the_missing_file(self):
        config = {"args": {"models": {"shape": "ckpts/model"}}}
        data = json.dumps(config).encode()
        assets = {
            "pipeline.json": ("pipeline.json", len(data), None),
            "ckpts/model.gguf": ("shape/model_Q8_0.gguf", 4, "ffff"),
        }
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(gguf_models, "RUNPOD_VOLUME", Path(temp)), \
             patch.object(gguf_models, "use_global_volume", return_value=True), \
             patch.object(gguf_models, "ARTIFACTS", {"shape/model": None}), \
             patch.object(gguf_models, "_artifacts", return_value=assets), \
             patch.object(gguf_models, "_stream_file") as stream:
            def download(url, destination, **kwargs):
                destination.write_bytes(data if url.endswith("pipeline.json") else b"gguf")
            stream.side_effect = download
            snapshot, actual = gguf_models.gguf_snapshot()
            self.assertEqual(actual, config)
            self.assertEqual(stream.call_count, 2)
            gguf_models.gguf_snapshot()
            self.assertEqual(stream.call_count, 2)
            (snapshot / "ckpts/model.gguf").write_bytes(b"x")
            gguf_models.gguf_snapshot()
            self.assertEqual(stream.call_count, 3)
            self.assertEqual(stream.call_args.kwargs["sha256"], "ffff")
            self.assertEqual(list(snapshot.glob("**/.download-*")), [])
