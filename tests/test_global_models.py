import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import handler
from scripts.global_models import NAF_CHECKPOINT, _stream_file, verify_global_models
from scripts.model_cache import configure_cache


class GlobalModelTests(unittest.TestCase):
    def test_writable_caches_are_local(self):
        with patch.dict(os.environ, {"PIXAL3D_GLOBAL_VOLUME": "1", "HF_HUB_CACHE": "/runpod-volume/hub"}, clear=True), \
             patch.object(Path, "mkdir"):
            configure_cache()
            for key in ("HF_HUB_CACHE", "HF_XET_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR", "FLEX_GEMM_AUTOTUNE_CACHE_PATH"):
                self.assertTrue(os.environ[key].startswith("/tmp/pixal3d-cache/"), key)

    def test_prefill_action_is_enabled_only_in_prefill_mode(self):
        job = {"input": {"action": "prefill_models"}}
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn("error", handler.handler(job))
        with patch.dict(os.environ, {"PIXAL3D_GLOBAL_VOLUME": "1", "PIXAL3D_PREFILL_ONLY": "1"}, clear=True), \
             patch("scripts.global_models.prefill_global_models", return_value={"status": "ready"}) as prefill:
            self.assertEqual(handler.handler(job), {"status": "ready"})
            self.assertIn("error", handler.handler({"input": {"image_url": "https://example.org/image"}}))
            prefill.assert_called_once_with()

    def test_prefill_writes_regular_files_and_rejects_truncated_data(self):
        fake = Mock(ok=True, headers={})
        fake.iter_content.return_value = [b"ab", b"c"]
        fake.__enter__ = Mock(return_value=fake)
        fake.__exit__ = Mock(return_value=False)
        get = Mock(return_value=fake)
        sdk = SimpleNamespace(get=get, RequestException=OSError)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"requests": sdk}):
            path = Path(directory) / "model.safetensors"
            with self.assertRaisesRegex(RuntimeError, "Incomplete download"):
                _stream_file("https://example.org/model", path, size=4)
            self.assertEqual(_stream_file("https://example.org/model", path, size=3), 3)
            self.assertFalse(path.is_symlink())
            get.assert_called_once()

    def test_manifest_requires_every_file_and_exact_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "pixal3d/pipeline.json": json.dumps({"args": {"models": {"decoder": "ckpts/decoder"}}}),
                "pixal3d/ckpts/decoder.json": "{}",
                "pixal3d/ckpts/decoder.safetensors": "weights",
                "moge/model.pt": "weights",
                "dinov3/config.json": "{}",
                "dinov3/preprocessor_config.json": "{}",
                "dinov3/model.safetensors": "weights",
                NAF_CHECKPOINT: "checkpoint",
            }
            for name, content in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(content)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                verify_global_models(root)
            sizes = {name: len(content) for name, content in files.items()}
            (root / "manifest.json").write_text(json.dumps(sizes))
            self.assertEqual(verify_global_models(root), sizes)
            (root / "moge/model.pt").write_text("short")
            with self.assertRaisesRegex(RuntimeError, "moge/model.pt"):
                verify_global_models(root)
