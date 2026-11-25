import io
import json
import os
from pathlib import Path
import struct
import unittest
from unittest.mock import Mock, patch

import export_handler as worker
from handler import handler
from scripts.asset_storage import asset_source_key


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class ExportWorkerTests(unittest.TestCase):
    def setUp(self):
        self.asset_id = "a" * 32
        self.inputs = {"action": "export_glb", "asset_id": self.asset_id}
        self.storage = Mock()
        env = patch.dict(os.environ, {"PIXAL3D_OUTPUT_BUCKET": "test-results"})
        client = patch.object(worker, "storage_client", return_value=self.storage)
        self.addCleanup(env.stop)
        self.addCleanup(client.stop)
        env.start()
        self.client = client.start()

    def test_validates_options_before_storage_use(self):
        for invalid in (
            {"asset_id": "../input.usdz"},
            {"asset_id": self.asset_id, "texture_size": 1000},
            {"asset_id": self.asset_id, "target_triangles": True},
        ):
            self.assertIn("error", handler({"input": {"action": "export_glb", **invalid}}))
        self.client.assert_not_called()

    def test_cached_variant_skips_blender_and_source_download(self):
        self.storage.head_object.return_value = {
            "ContentLength": 345, "Metadata": {"triangle-count": "4", "vertex-count": "6"},
        }
        self.storage.generate_presigned_url.return_value = "https://storage.example/asset.glb"
        with patch.object(worker.subprocess, "run") as blender:
            result = handler({"input": self.inputs})
        self.assertTrue(result["cached"])
        self.assertEqual(result["triangle_count"], 4)
        self.assertEqual(result["glb_url"], self.storage.generate_presigned_url.return_value)
        blender.assert_not_called()
        self.storage.get_object.assert_not_called()

    def test_blender_worker_reads_source_and_uploads_rendered_variant(self):
        self.storage.head_object.side_effect = MissingObject
        self.storage.get_object.return_value = {"ContentLength": 4, "Body": io.BytesIO(b"USDZ")}
        self.storage.generate_presigned_url.return_value = "https://storage.example/asset.glb"
        mesh = {
            "accessors": [{"count": 6}, {"count": 12}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        }
        metadata = json.dumps(mesh).encode()
        content = struct.pack("<4sIII4s", b"glTF", 2, 20 + len(metadata), len(metadata), b"JSON") + metadata

        def run(command, **options):
            self.assertEqual(Path(command[-4]).read_bytes(), b"USDZ")
            self.assertEqual(command[-2:], ["15000", "1024"])
            Path(command[-3]).write_bytes(content)

        with patch.object(worker.subprocess, "run", side_effect=run) as blender:
            result = handler({"input": {**self.inputs, "target_triangles": 15000, "texture_size": 1024}})
        blender.assert_called_once()
        self.storage.get_object.assert_called_once_with(Bucket="test-results", Key=asset_source_key(self.asset_id))
        self.assertFalse(result["cached"])
        self.assertEqual(result["triangle_count"], 4)
        self.assertEqual(result["vertex_count"], 6)
        self.assertEqual(result["size_bytes"], len(content))
        self.assertIn("/" + self.asset_id + "/", self.storage.upload_file.call_args.args[2])
        self.assertEqual(self.storage.upload_file.call_args.kwargs["ExtraArgs"]["Metadata"]["triangle-count"], "4")


if __name__ == "__main__":
    unittest.main()
