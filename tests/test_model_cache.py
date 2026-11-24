import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.model_cache import pipeline_snapshot


class PipelineSnapshotTests(unittest.TestCase):
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
