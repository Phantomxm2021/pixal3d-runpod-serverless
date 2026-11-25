import base64
import io
import os
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import handler as worker


class RunpodWorkerTests(unittest.TestCase):
    def test_source_entrypoint_registers_handler(self):
        start = Mock()
        sdk = SimpleNamespace(serverless=SimpleNamespace(start=start))
        stdout = io.StringIO()
        with patch.dict(sys.modules, {"runpod": sdk}), \
             patch.dict(os.environ, {"PIXAL3D_STAGE_TRACE": "1"}, clear=True), \
             patch("faulthandler.dump_traceback_later") as arm, \
             patch("faulthandler.cancel_dump_traceback_later") as disarm, redirect_stdout(stdout):
            runpy.run_module("rp_handler", run_name="__main__")
            handler = start.call_args.args[0]["handler"]
            self.assertIn("error", handler({"id": "test-job", "input": {}}))
        start.assert_called_once()
        self.assertEqual(arm.call_count, 2)
        disarm.assert_called_once()
        markers = stdout.getvalue().splitlines()
        self.assertIn("Python entrypoint reached", markers[0])
        self.assertIn("Importing Runpod SDK", markers[1])
        self.assertIn("Runpod SDK imported", markers[2])
        self.assertIn("Registering handler", markers[3])
        self.assertEqual(markers[4], "[request test-job] Entered handler")
        self.assertIn("Handler returned", markers[5])

    def test_base64_input_and_output(self):
        def infer(source, output, *, seed, skip_rembg, target_triangles, texture_size):
            self.assertEqual(source.read_bytes(), b"image")
            self.assertEqual(source.suffix, ".jpg")
            self.assertEqual(seed, 5)
            self.assertTrue(skip_rembg)
            self.assertEqual((target_triangles, texture_size), (200_000, 2048))
            output.write_bytes(b"glTF")
            return {"inference_seconds": 1}, {"triangle_count": 200, "vertex_count": 150}

        with patch.object(worker, "_run_inference", side_effect=infer):
            result = worker.handler({"input": {
                "image_base64": "data:image/jpeg;base64,aW1hZ2U=",
                "suffix": "jpg",
                "seed": "5",
                "skip_rembg": True,
            }})
        self.assertEqual(base64.b64decode(result["glb_base64"]), b"glTF")
        self.assertEqual(result["size_bytes"], 4)
        self.assertEqual(result["triangle_count"], 200)
        self.assertEqual(result["vertex_count"], 150)

    def test_direct_glb_passes_requested_bake_options_to_inference(self):
        def infer(source, output, *, seed, skip_rembg, target_triangles, texture_size):
            self.assertEqual(output.suffix, ".glb")
            self.assertEqual((target_triangles, texture_size), (30_000, 1024))
            output.write_bytes(b"glTF")
            return {}, {"triangle_count": 29_998, "vertex_count": 15_000}

        inference = SimpleNamespace(run_inference=Mock(side_effect=infer), MODEL_LOAD_TIMINGS={})
        with patch.dict(sys.modules, {"inference_worker": inference}):
            result = worker.handler({"input": {
                "image_base64": "aW1hZ2U=",
                "target_triangles": 30_000,
                "texture_size": 1024,
            }})

        inference.run_inference.assert_called_once()
        self.assertEqual(result["target_triangles"], 30_000)
        self.assertEqual(result["triangle_count"], 29_998)
        self.assertEqual(result["texture_size"], 1024)

    def test_oversized_glb_can_be_uploaded(self):
        def infer(source, output, **kwargs):
            self.assertEqual(source.read_bytes(), b"source")
            output.write_bytes(b"large glb")
            return {"inference_seconds": 1}, {"triangle_count": 200, "vertex_count": 150}

        download = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(b"source"))
        upload = Mock(return_value="https://storage.example/model.glb")
        inputs = {"image_url": "https://example.com/input.png"}
        with patch.object(worker, "urlopen", download), patch.object(worker, "_upload_glb", upload), \
             patch.object(worker, "_run_inference", side_effect=infer), \
             patch.object(worker, "MAX_INLINE_GLB_BYTES", 4), \
             patch.dict(os.environ, {}, clear=True):
            self.assertIn("exceeds", worker.handler({"input": inputs})["error"])
            upload.assert_not_called()
            with patch.dict(os.environ, {"PIXAL3D_OUTPUT_BUCKET": "results"}):
                strict_inline = worker.handler({"input": {**inputs, "output_format": "base64"}})
                self.assertIn("exceeds", strict_inline["error"])
                upload.assert_not_called()
                for output_format in ("auto", "url"):
                    result = worker.handler({"input": {**inputs, "output_format": output_format}})
                    self.assertEqual(result["glb_url"], upload.return_value)
                    self.assertEqual(result["size_bytes"], 9)
                    self.assertIn("upload_seconds", result["timing"])
                result = worker.handler({"input": inputs})
        self.assertEqual(upload.call_count, 3)
        self.assertEqual(upload.call_args.args[1], "results")
        self.assertFalse(upload.call_args.args[0].exists())
        self.assertEqual(result["glb_url"], upload.return_value)
        self.assertEqual(result["triangle_count"], 200)

    def test_auto_keeps_small_glb_inline_when_bucket_is_configured(self):
        def infer(source, output, **kwargs):
            output.write_bytes(b"glTF")
            return {}, {"triangle_count": 10, "vertex_count": 6}

        with patch.object(worker, "_run_inference", side_effect=infer), \
             patch.object(worker, "_upload_glb") as upload, \
             patch.dict(os.environ, {"PIXAL3D_OUTPUT_BUCKET": "results"}, clear=True):
            result = worker.handler({"input": {"image_base64": "aW1hZ2U="}})
        self.assertEqual(base64.b64decode(result["glb_base64"]), b"glTF")
        upload.assert_not_called()

    def test_usdz_publishes_the_baked_asset_and_id(self):
        def infer(source, output, **kwargs):
            self.assertEqual(output.suffix, ".usdz")
            output.write_bytes(b"USDZ")
            return {}, {"quad_count": 2, "vertex_count": 6}

        with patch.object(worker, "_run_inference", side_effect=infer), \
             patch.object(worker, "_upload_usdz", return_value={
                 "asset_id": "a" * 32, "usdz_url": "https://storage.example/source.usdz",
             }) as upload, patch.dict(os.environ, {"PIXAL3D_OUTPUT_BUCKET": "results"}):
            result = worker.handler({"input": {"image_base64": "aW1hZ2U=", "asset_format": "usdz"}})
        self.assertEqual(result["quad_count"], 2)
        self.assertEqual(result["mime_type"], "model/vnd.usdz+zip")
        self.assertEqual(result["asset_id"], "a" * 32)
        self.assertEqual(result["usdz_url"], "https://storage.example/source.usdz")
        self.assertEqual(upload.call_args.args[1], "results")

    def test_invalid_jobs_do_not_start_inference(self):
        invalid = [
            None,
            {"image_url": "file:///input.png"},
            {"image_base64": "%%%"},
            {"image_base64": "aW1hZ2U=", "suffix": "../../test"},
            {"image_base64": "aW1hZ2U=", "skip_rembg": "false"},
            {"image_base64": "aW1hZ2U=", "output_format": "url"},
            {"image_base64": "aW1hZ2U=", "target_triangles": 999},
            {"image_base64": "aW1hZ2U=", "target_triangles": 200_001},
            {"image_base64": "aW1hZ2U=", "target_triangles": True},
            {"image_base64": "aW1hZ2U=", "texture_size": 1536},
            {"image_base64": "aW1hZ2U=", "texture_size": 1024.0},
            {"image_base64": "aW1hZ2U=", "asset_format": "usdz"},
            {"image_base64": "aW1hZ2U=", "asset_format": "usdz", "output_format": "base64"},
        ]
        with patch.object(worker, "_run_inference") as infer, patch.dict(os.environ, {}, clear=True):
            for inputs in invalid:
                with self.subTest(inputs=inputs):
                    self.assertIn("error", worker.handler({"input": inputs}))
        infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
