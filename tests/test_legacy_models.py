import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import legacy_models


class LegacyModelRetirementTests(unittest.TestCase):
    def test_removes_only_legacy_files_after_checking_the_gguf_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            volume = Path(temp).resolve()
            global_model_root = volume / "pixal3d-models-v1"
            old_file = volume / "hi3d-models-v1/hi3d/vae/model.safetensors"
            old_pixal = global_model_root / "pixal3d/pipeline.json"
            needed = global_model_root / "dinov3/model.safetensors"
            gguf = volume / "pixal3d-gguf-q8-v1/ckpts/shape.gguf"
            for file in (old_file, old_pixal, needed, gguf):
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"x")
            unknown = old_file.parent / "other.bin"
            unknown.write_bytes(b"x")
            client = Mock()
            key = f"pixal3d/{'a' * 32}.glb"
            job = {"validated_glb_key": key}

            with patch.object(legacy_models, "RUNPOD_VOLUME", volume), \
                 patch.object(legacy_models, "GLOBAL_MODEL_ROOT", global_model_root), \
                 patch.object(Path, "is_mount", return_value=True), \
                 patch.dict(os.environ, {"PIXAL3D_OUTPUT_BUCKET": "bucket", "PIXAL3D_GLOBAL_VOLUME": "1"}), \
                 patch.object(legacy_models, "storage_client", return_value=client):
                client.head_object.return_value = {"ContentType": "model/gltf-binary", "ContentLength": 2}
                self.assertIn("error", legacy_models.retire_legacy_model_cache({**job, "apply": True}))
                self.assertTrue(old_file.exists())
                client.head_object.return_value["Metadata"] = {
                    "model-format": legacy_models.MODEL_FORMAT, "model-revision": legacy_models.REVISION,
                }
                self.assertEqual(legacy_models.retire_legacy_model_cache(job)["files"], 2)
                self.assertTrue(old_file.exists())
                self.assertEqual(legacy_models.retire_legacy_model_cache({**job, "apply": True}), {
                    "status": "retired", "files": 2, "model_bytes": 2,
                })
            client.head_object.assert_called_with(Bucket="bucket", Key=key)
            self.assertFalse(old_file.exists())
            self.assertFalse(old_pixal.exists())
            self.assertTrue(needed.exists())
            self.assertTrue(gguf.exists())
            self.assertTrue(unknown.exists())
