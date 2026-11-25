import io
import json
import os
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import hi3d_handler as worker
from handler import handler
from scripts.asset_storage import white_mesh_key


def _glb(triangles: int) -> bytes:
    document = {
        "accessors": [{"count": 8}, {"count": triangles * 3}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
    }
    metadata = json.dumps(document).encode()
    metadata += b" " * (-len(metadata) % 4)
    return struct.pack("<4sIII4s", b"glTF", 2, 20 + len(metadata), len(metadata), b"JSON") + metadata


class Hi3DHandlerTests(unittest.TestCase):
    def setUp(self):
        self.asset_id = "f" * 32
        self.inputs = {
            "action": "refine_white_mesh", "asset_id": self.asset_id,
            "image_base64": "aW1hZ2U=", "seed": -5, "skip_rembg": True,
        }
        self.env = patch.dict(os.environ, {
            "PIXAL3D_OUTPUT_BUCKET": "test-assets", "HI3D_PYTHON": sys.executable,
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_rejects_invalid_refinement_before_storage_and_gpu(self):
        with patch.object(worker, "refine_white_mesh") as refine:
            for invalid in ({"asset_id": "../file"}, {"resolution": 1536}, {"num_inference_steps": False},
                            {"output_format": "base64"}, {"target_triangles": 10}):
                self.assertIn("error", handler({"input": {**self.inputs, **invalid}}))
        refine.assert_not_called()

    def test_cache_verification_fails_before_using_gpu(self):
        with patch.object(worker, "verify_hi3d_models", side_effect=FileNotFoundError("missing")), \
             patch.object(worker.subprocess, "run") as run:
            result = handler({"input": self.inputs})
        self.assertIn("prepare_hi3d_cache", result["error"])
        run.assert_not_called()

    def test_refinement_uses_saved_mesh_and_publishes_a_new_glb(self):
        source = _glb(12)
        refined = _glb(20)
        storage = Mock()
        storage.get_object.return_value = {"ContentLength": len(source), "Body": io.BytesIO(source)}
        storage.generate_presigned_url.return_value = "https://example.com/refined.glb"
        release = Mock()

        def run(command, **kwargs):
            release.assert_called_once_with()
            self.assertEqual(command[:4], [sys.executable, "-u", "-m", "scripts.hi3d_worker"])
            self.assertEqual(Path(command[command.index("--source") + 1]).read_bytes(), source)
            self.assertEqual(Path(command[command.index("--image") + 1]).read_bytes(), b"image")
            self.assertEqual(command[command.index("--resolution") + 1], "512")
            self.assertIn("--seed=-5", command)
            self.assertIn("--skip-rembg", command)
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            Path(command[command.index("--output") + 1]).write_bytes(refined)
            Path(command[command.index("--report") + 1]).write_text(json.dumps({
                "mesh_resolution": 512, "peak_vram_allocated_bytes": 400, "timing": {"hi3d_refinement_seconds": 1.5},
            }))

        with patch.object(worker, "verify_hi3d_models"), \
             patch.object(worker, "storage_client", return_value=storage), \
             patch.dict(sys.modules, {"inference_worker": SimpleNamespace(release_models=release)}), \
             patch.object(worker.subprocess, "run", side_effect=run):
            result = handler({"input": self.inputs})

        storage.get_object.assert_called_once_with(Bucket="test-assets", Key=white_mesh_key(self.asset_id))
        self.assertNotEqual(result["asset_id"], self.asset_id)
        self.assertEqual(storage.upload_file.call_args.args[1:], ("test-assets", white_mesh_key(result["asset_id"])))
        self.assertEqual(result["source_asset_id"], self.asset_id)
        self.assertEqual(result["source_triangle_count"], 12)
        self.assertEqual(result["triangle_count"], 20)
        self.assertEqual(result["timing"]["hi3d_refinement_seconds"], 1.5)
        self.assertEqual(result["peak_vram_allocated_bytes"], 400)


if __name__ == "__main__":
    unittest.main()
