import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.model_cache import pipeline_snapshot, runpod_pipeline_snapshot


class PipelineSnapshotTests(unittest.TestCase):
    def test_runpod_snapshot_uses_refs_and_requires_checkpoint_blobs(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory)
            repo = hub / "models--TencentARC--Pixal3D"
            revision = "a" * 40
            snapshot = repo / "snapshots" / revision
            (snapshot / "ckpts").mkdir(parents=True)
            (repo / "refs").mkdir()
            (repo / "refs/main").write_text(revision + "\n", encoding="utf-8")
            config = {"args": {"models": {"decoder": "ckpts/decoder"}}}
            (snapshot / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
            (snapshot / "ckpts/decoder.json").write_text("{}", encoding="utf-8")
            blob = repo / "blobs/weights"
            blob.parent.mkdir()
            blob.write_bytes(b"weights")
            (snapshot / "ckpts/decoder.safetensors").symlink_to(blob)
            other = repo / "snapshots" / ("b" * 40)
            other.mkdir()
            with patch("scripts.model_cache.RUNPOD_MODEL_HUB", hub):
                self.assertEqual(runpod_pipeline_snapshot(), (snapshot, config))
                (repo / "refs/main").unlink()
                with self.assertRaisesRegex(RuntimeError, "Expected one snapshot"):
                    runpod_pipeline_snapshot()
                other.rmdir()
                self.assertEqual(runpod_pipeline_snapshot(), (snapshot, config))
                blob.unlink()
                with self.assertRaisesRegex(RuntimeError, "decoder.safetensors"):
                    runpod_pipeline_snapshot()

    def test_only_downloads_configured_single_image_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "commit"
            snapshot.mkdir()
            config = {"args": {"models": {"decoder": "ckpts/decoder"}}}
            config_path = snapshot / "pipeline.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            sdk = SimpleNamespace(
                hf_hub_download=Mock(return_value=str(config_path)),
                snapshot_download=Mock(return_value=str(snapshot)),
            )
            with patch.dict(sys.modules, {"huggingface_hub": sdk}):
                self.assertEqual(pipeline_snapshot("owner/model"), (snapshot, config))
                sdk.snapshot_download.assert_called_once_with(
                    "owner/model", revision="commit",
                    allow_patterns=["ckpts/decoder.json", "ckpts/decoder.safetensors"],
                )
                (snapshot / "ckpts").mkdir()
                for filename in ("decoder.json", "decoder.safetensors"):
                    (snapshot / "ckpts" / filename).touch()
                sdk.snapshot_download.reset_mock()
                self.assertEqual(pipeline_snapshot("owner/model"), (snapshot, config))
                sdk.snapshot_download.assert_not_called()
