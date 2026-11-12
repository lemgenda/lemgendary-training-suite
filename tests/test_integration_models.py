"""Integration tests for model docs generation, variation staging, and FastAPI endpoints."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

from training.cloud.metadata_generator import KaggleMetadataService
from training.cloud.model_metadata import ModelMetadata
from training.doc_generator import ModelDocumentationService
from training.export.onnx_weights import WeightLayout, save_proto
from training.export.staging import VariationStagingService
from training.export.validation import StrictIsolationValidator
from training.server.app import create_app
from training.utils.paths import get_project_root


def _tiny_onnx_model() -> onnx.ModelProto:
    """``Y = X + W`` with a 16-element FP32 initializer ``W``."""
    weight = numpy_helper.from_array(np.arange(16, dtype=np.float32), name="W")
    graph = helper.make_graph(
        [helper.make_node("Add", ["X", "W"], ["Y"])],
        "tiny",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [16])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [16])],
        initializer=[weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


class TestIntegrationModels(unittest.TestCase):
    """Verifies the end-to-end model lifecycle: docs, variation staging, and FastAPI routes."""

    def setUp(self) -> None:
        self.project_root = get_project_root()
        self.app = create_app(project_root=self.project_root, enforce_auth=False)
        self.client = TestClient(self.app)

    def test_e2e_doc_and_metadata_generation(self) -> None:
        """Verify end-to-end documentation and Kaggle metadata generation for a model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "LemGendaryModels" / "mirnet_exposure"
            out_dir.mkdir(parents=True)

            doc_service = ModelDocumentationService()
            meta_service = KaggleMetadataService(owner_slug="lemtrada")

            readme_file = doc_service.write_readme("mirnet_exposure", output_dir=out_dir)
            meta_file = meta_service.write_metadata("mirnet_exposure", output_dir=out_dir)

            self.assertTrue(readme_file.exists())
            self.assertTrue(meta_file.exists())

            # Verify README content
            readme_text = readme_file.read_text(encoding="utf-8")
            self.assertIn("MIRNet", readme_text)
            self.assertIn("Scientific Research", readme_text)

            # Verify metadata content
            self.assertEqual(meta_file.name, "model_metadata.yaml")
            metadata = ModelMetadata.load(meta_file)
            self.assertEqual(metadata.model.publishing.license, "CC BY-NC 4.0")
            self.assertFalse(metadata.model.publishing.is_private)

            json_meta_file = out_dir / "model-metadata.json"
            self.assertTrue(json_meta_file.exists())
            with open(json_meta_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            self.assertEqual(json_data["licenseName"], "CC BY-NC 4.0")
            self.assertFalse(json_data["isPrivate"])

    def test_e2e_variation_staging_and_isolation(self) -> None:
        """Verify end-to-end variation staging and strict isolation audit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "LemGendaryModels" / "mirnet_exposure"
            model_dir.mkdir(parents=True)

            # Create mock source exports and checkpoints
            src_pt = model_dir / "LemGendaryMirNetExposure.pt"
            src_pt.write_bytes(b"dummy_pt_weights")
            # A real graph with a sidecar: staging renames it and must rewrite the location.
            src_fp32 = model_dir / "LemGendaryMirNetExposure_FP32.onnx"
            save_proto(_tiny_onnx_model(), src_fp32, WeightLayout.SIDECAR)
            src_fp16 = model_dir / "LemGendaryMirNetExposure.onnx"
            src_fp16.write_bytes(b"dummy_fp16_onnx")

            ckpt_dir = model_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "best.pth").write_bytes(b"dummy_best_checkpoint")

            staging_svc = VariationStagingService()
            staging_res = staging_svc.stage_variations(
                model_key="mirnet_exposure",
                source_files={
                    "pytorch": src_pt,
                    "onnx_fp32": src_fp32,
                    "onnx_fp16": src_fp16,
                },
                base_dir=model_dir,
                validate=True,
            )

            self.assertTrue(staging_res.success)
            assert staging_res.report is not None
            self.assertTrue(staging_res.report.is_valid)

            # Audit with validator
            validator = StrictIsolationValidator()
            report = validator.validate_all("mirnet_exposure", base_dir=model_dir)
            self.assertTrue(report.is_valid)
            self.assertEqual(len(report.variations), 4)

            # The renamed FP32 graph loads its weights from the renamed sidecar.
            staged_fp32 = staging_res.staged_files["onnx_fp32"]
            self.assertEqual(
                staging_res.staged_files["onnx_fp32_data"].name, f"{staged_fp32.name}.data"
            )
            reloaded = onnx.load(str(staged_fp32))
            np.testing.assert_array_equal(
                numpy_helper.to_array(reloaded.graph.initializer[0]),
                np.arange(16, dtype=np.float32),
            )

    def test_fastapi_docs_generate_endpoint(self) -> None:
        """Verify POST /api/models/{model_key}/docs/generate."""
        resp = self.client.post("/api/models/mirnet_exposure/docs/generate")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model_key"], "mirnet_exposure")
        self.assertIn("readme_path", data)
        self.assertIn("metadata_path", data)

    def test_fastapi_variations_validate_endpoint(self) -> None:
        """Verify GET /api/models/{model_key}/variations/validate."""
        resp = self.client.get("/api/models/mirnet_exposure/variations/validate")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_key"], "mirnet_exposure")
        self.assertIn("is_valid", data)
        self.assertIn("variations", data)

    def test_fastapi_kaggle_push_model_dry_run(self) -> None:
        """Verify POST /api/training/kaggle/push-model with dry_run=True."""
        payload = {
            "model": "mirnet_exposure",
            "variation": "all",
            "dry_run": True,
            "version_notes": "Integration test dry run release",
        }
        resp = self.client.post("/api/training/kaggle/push-model", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("model_key"), "mirnet_exposure")
        self.assertIn("variations", data)
        for var_res in data["variations"].values():
            self.assertTrue(var_res.get("dry_run", False))


if __name__ == "__main__":
    unittest.main()
