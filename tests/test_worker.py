import base64
import io
import os
import runpy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import handler as worker


class RunpodWorkerTests(unittest.TestCase):
    def test_source_entrypoint_registers_handler(self):
        start = Mock()
        sdk = SimpleNamespace(serverless=SimpleNamespace(start=start))
        with patch.dict(sys.modules, {"runpod": sdk}), patch.dict(os.environ, {}, clear=True):
            runpy.run_module("rp_handler", run_name="__main__")
        start.assert_called_once()
        handler = start.call_args.args[0]["handler"]
        self.assertIn("error", handler({"input": {}}))

    def test_base64_input_and_output(self):
        def infer(source, output, *, seed, skip_rembg):
            self.assertEqual(source.read_bytes(), b"image")
            self.assertEqual(source.suffix, ".jpg")
            self.assertEqual(seed, 5)
            self.assertTrue(skip_rembg)
            output.write_bytes(b"glTF")
            return {"inference_seconds": 1}

        with patch.object(worker, "_run_inference", side_effect=infer):
            result = worker.handler({"input": {
                "image_base64": "data:image/jpeg;base64,aW1hZ2U=",
                "suffix": "jpg",
                "seed": "5",
                "skip_rembg": True,
            }})
        self.assertEqual(base64.b64decode(result["glb_base64"]), b"glTF")
        self.assertEqual(result["size_bytes"], 4)

    def test_oversized_glb_can_be_uploaded(self):
        def infer(source, output, **kwargs):
            self.assertEqual(source.read_bytes(), b"source")
            output.write_bytes(b"large glb")
            return {"inference_seconds": 1}

        download = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(b"source"))
        upload = Mock(return_value="https://storage.example/model.glb")
        inputs = {"image_url": "https://example.com/input.png"}
        with patch.object(worker, "urlopen", download), patch.object(worker, "_upload_glb", upload), \
             patch.object(worker, "_run_inference", side_effect=infer), \
             patch.object(worker, "MAX_INLINE_GLB_BYTES", 4), \
             patch.dict(os.environ, {"PIXAL3D_OUTPUT_BUCKET": "results"}):
            self.assertIn("exceeds", worker.handler({"input": inputs})["error"])
            result = worker.handler({"input": {**inputs, "output_format": "url"}})
        upload.assert_called_once()
        self.assertEqual(upload.call_args.args[1], "results")
        self.assertFalse(upload.call_args.args[0].exists())
        self.assertEqual(result["glb_url"], upload.return_value)

    def test_invalid_jobs_do_not_start_inference(self):
        invalid = [
            None,
            {"image_url": "file:///input.png"},
            {"image_base64": "%%%"},
            {"image_base64": "aW1hZ2U=", "suffix": "../../test"},
            {"image_base64": "aW1hZ2U=", "skip_rembg": "false"},
            {"image_base64": "aW1hZ2U=", "output_format": "url"},
        ]
        with patch.object(worker, "_run_inference") as infer, patch.dict(os.environ, {}, clear=True):
            for inputs in invalid:
                with self.subTest(inputs=inputs):
                    self.assertIn("error", worker.handler({"input": inputs}))
        infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
