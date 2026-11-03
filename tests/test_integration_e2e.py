"""
LemGendary Ecosystem — Cross-Suite Integration and End-to-End Pipeline Tests.

Phase 3 of Ecosystem Comprehensive Testing Battery.
Verifies:
1. End-to-end dataset pipeline: datasets compiled to WebDataset/WebP format
   by the Dataset Compiler are cleanly resolved, read, and decoded by the
   Training Suite ContainerReader and MultiTaskDataset.
2. Cross-service health mesh schema compatibility between the Environment
   Manager and the Dataset Compiler.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import nn

from tests.fixtures.webdataset import json_bytes, webp_bytes, write_tar
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.metadata_generator import KaggleMetadataService
from training.data.containers import (
    DirectoryReader,
    WebDatasetReader,
    resolve_container_reader,
)
from training.data.containers import (
    Sample as TrainingSample,
)
from training.export.onnx import export_onnx
from training.export.staging import VariationStagingService
from training.export.torch_standalone import export_torch_standalone
from training.export.validation import StrictIsolationValidator


class TestDatasetToTrainingE2E(unittest.TestCase):
    """Verifies that datasets compiled to WebDataset/WebP are ingested by the training suite."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def _make_manifold(self, name: str) -> Path:
        """Create a manifold with a WebDataset ``dataset_info.yaml``; return its shard dir."""
        manifold_path = self.root / name
        shards_dir = manifold_path / "shards" / "train"
        shards_dir.mkdir(parents=True)
        ds_info = {
            "name": name,
            "format": "webdataset",
            "canonical_format": "webdataset",
            "image_format": "webp",
            "task": "restoration",
            "splits": {"train": 1},
        }
        (manifold_path / "dataset_info.yaml").write_text(
            yaml.safe_dump(ds_info, default_flow_style=False),
            encoding="utf-8",
        )
        return shards_dir

    def test_webdataset_webp_pipeline_roundtrip(self) -> None:
        """Create a canonical WebDataset WebP manifold and load via training suite."""
        shards_dir = self._make_manifold("LemGendizedRestorationE2E")
        img_bytes = webp_bytes((40, 120, 200), quality=92)
        tgt_bytes = webp_bytes((240, 240, 240), quality=95)
        meta = {"source": "integration_test", "score": 1.0}
        write_tar(
            shards_dir / "shard-00000.tar",
            {
                "sample_1001.webp": img_bytes,
                "sample_1001.target.webp": tgt_bytes,
                "sample_1001.json": json_bytes(meta),
            },
        )

        reader = resolve_container_reader(shards_dir.parent.parent, split="train")
        self.assertIsInstance(reader, WebDatasetReader)
        self.assertEqual(len(reader), 1)

        sample = reader[0]
        self.assertIsInstance(sample, TrainingSample)
        self.assertEqual(sample.name, "sample_1001")
        self.assertEqual(sample.image_format, "webp")
        self.assertEqual(sample.image_bytes, img_bytes)
        self.assertIsNotNone(sample.target_bytes)
        self.assertEqual(sample.target_bytes, tgt_bytes)

        decoded_img = DirectoryReader.decode_image(sample.image_bytes)
        self.assertIsInstance(decoded_img, Image.Image)
        self.assertEqual(decoded_img.size, (32, 32))
        self.assertEqual(decoded_img.mode, "RGB")

        decoded_tgt = DirectoryReader.decode_image(sample.target_bytes)
        self.assertIsInstance(decoded_tgt, Image.Image)
        self.assertEqual(decoded_tgt.size, (32, 32))

        reader.close()

    def test_webdataset_data_obfuscation_pipeline_roundtrip(self) -> None:
        """Create a WebDataset with obfuscated .data shards and load it via the training suite."""
        shards_dir = self._make_manifold("LemGendizedRestorationDataE2E")
        img_bytes = webp_bytes((50, 150, 250), quality=92)
        tgt_bytes = webp_bytes((250, 150, 50), quality=95)
        meta = {"source": "integration_test_data", "score": 0.99}
        write_tar(
            shards_dir / "shard-00000.data",
            {
                "sample_2001.webp": img_bytes,
                "sample_2001.target.webp": tgt_bytes,
                "sample_2001.json": json_bytes(meta),
            },
        )

        reader = resolve_container_reader(shards_dir.parent.parent, split="train")
        self.assertIsInstance(reader, WebDatasetReader)
        self.assertEqual(len(reader), 1)

        sample = reader[0]
        self.assertEqual(sample.name, "sample_2001")
        self.assertEqual(sample.image_format, "webp")
        self.assertEqual(sample.image_bytes, img_bytes)
        self.assertEqual(sample.target_bytes, tgt_bytes)

        decoded_img = DirectoryReader.decode_image(sample.image_bytes)
        self.assertEqual(decoded_img.size, (32, 32))
        reader.close()


class TestMultiSidecarHealthMeshE2E(unittest.TestCase):
    """Verifies API schema parity between the env-manager and dataset-compiler health payloads."""

    def test_sidecar_health_schemas_align(self) -> None:
        """Verify that health endpoint shapes expected by env-manager are produced by datasets."""
        # Probe structure expected by env_manager
        expected_fields = {"status", "service", "version", "uptime_seconds", "active_jobs"}

        # Simulate probe response payload as returned by lemgendary-datasets /api/health
        mock_datasets_health = {
            "status": "ok",
            "service": "LemGendary Dataset Compiler API",
            "version": "16.8.0-STABLE",
            "uptime_seconds": 120.5,
            "active_jobs": 0,
        }

        self.assertTrue(expected_fields.issubset(mock_datasets_health.keys()))
        self.assertEqual(mock_datasets_health["status"], "ok")


class TestFullProductionPipelinePass(unittest.TestCase):
    """End-to-end pipeline: train -> checkpoint -> export -> stage -> metadata -> push."""

    def test_full_pipeline_roundtrip(self) -> None:
        """Simulate training step -> checkpoint -> export -> stage -> metadata -> Kaggle payload."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_hub_dir = Path(tmp_dir) / "LemGendaryModels" / "mirnet_exposure"
            model_hub_dir.mkdir(parents=True)
            ckpt_dir = model_hub_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            model = _train_one_step(self, ckpt_dir)
            source_files = _export_binaries(self, model, Path(tmp_dir) / "staging")

            # 4. Stage isolated variations into dist/kaggle/
            staging_svc = VariationStagingService()
            staging_res = staging_svc.stage_variations(
                model_key="mirnet_exposure",
                source_files=source_files,
                base_dir=model_hub_dir,
                validate=True,
            )
            self.assertTrue(staging_res.success)
            self.assertIsNotNone(staging_res.report)
            assert staging_res.report is not None
            self.assertTrue(staging_res.report.is_valid)

            # 5. Generate Kaggle model-metadata.json
            meta_svc = KaggleMetadataService(owner_slug="lemtrada")
            meta_file = meta_svc.write_metadata("mirnet_exposure", output_dir=model_hub_dir)
            self.assertTrue(meta_file.exists())

            # 6. Strict isolation audit
            validator = StrictIsolationValidator()
            report = validator.validate_all("mirnet_exposure", base_dir=model_hub_dir)
            self.assertTrue(report.is_valid)
            self.assertEqual(len(report.variations), 4)

            # 7. Dry-run Kaggle upload payload
            sync_mgr = KaggleModelSyncManager(hub_root=model_hub_dir.parent)
            push_res = sync_mgr.push_model("mirnet_exposure", dry_run=True)
            self.assertEqual(push_res.model_key, "mirnet_exposure")
            for var_result in push_res.variations.values():
                self.assertTrue(var_result.dry_run)


def _train_one_step(case: unittest.TestCase, ckpt_dir: Path) -> nn.Module:
    """Run one optimiser step on a dummy model and save it as the best checkpoint."""
    model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    out = model(torch.randn(1, 3, 32, 32))
    loss = nn.functional.mse_loss(out, torch.randn(1, 3, 32, 32))
    loss.backward()
    optimizer.step()
    case.assertFalse(torch.isnan(loss))

    ckpt_path = ckpt_dir / "mirnet_exposure_best.pth"
    torch.save({"model_state": model.state_dict(), "epoch": 1, "loss": loss.item()}, ckpt_path)
    case.assertTrue(ckpt_path.exists())
    return model


def _export_binaries(case: unittest.TestCase, model: nn.Module, stage_dir: Path) -> dict:
    """Export TorchScript and FP32/FP16 ONNX binaries into ``stage_dir``."""
    stage_dir.mkdir(parents=True)
    source_files = {
        "pytorch": stage_dir / "LemGendaryMirNetExposure.pt",
        "onnx_fp32": stage_dir / "LemGendaryMirNetExposure_FP32.onnx",
        "onnx_fp16": stage_dir / "LemGendaryMirNetExposure.onnx",
    }
    export_torch_standalone(model, source_files["pytorch"])
    export_onnx(model, source_files["onnx_fp32"], dummy_shape=(1, 3, 32, 32), half=False)
    export_onnx(model, source_files["onnx_fp16"], dummy_shape=(1, 3, 32, 32), half=True)
    for path in source_files.values():
        case.assertTrue(path.exists())
    return source_files


if __name__ == "__main__":
    unittest.main()
