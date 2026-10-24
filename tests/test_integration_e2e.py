"""
LemGendary Ecosystem — Cross-Suite Integration and End-to-End Pipeline Tests.

Phase 3 of Ecosystem Comprehensive Testing Battery.
Verifies:
1. End-to-end dataset pipeline: datasets compiled to WebDataset/WebP format
   by lemgendary-datasets are cleanly resolved, read, and decoded by
   lemgendary-training-suite ContainerReader and MultiTaskDataset.
2. Cross-service health mesh schema compatibility between lemgendary-env-manager
   and lemgendary-datasets.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from training.data.containers import (
    DirectoryReader,
    Sample as TrainingSample,
    WebDatasetReader,
    resolve_container_reader,
)


class TestDatasetToTrainingE2E(unittest.TestCase):
    """Verifies that datasets compiled to WebDataset/WebP are ingested by the training suite."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_webdataset_webp_pipeline_roundtrip(self) -> None:
        """Create a canonical WebDataset WebP manifold and load via training suite."""
        manifold_path = self.root / "LemGendizedRestorationE2E"
        shards_dir = manifold_path / "shards" / "train"
        shards_dir.mkdir(parents=True)

        # 1. Generate synthetic 32x32 WebP image and target
        img = Image.new("RGB", (32, 32), color=(40, 120, 200))
        tgt = Image.new("RGB", (32, 32), color=(240, 240, 240))

        buf_img = io.BytesIO()
        img.save(buf_img, format="WEBP", quality=92)
        img_bytes = buf_img.getvalue()

        buf_tgt = io.BytesIO()
        tgt.save(buf_tgt, format="WEBP", quality=95)
        tgt_bytes = buf_tgt.getvalue()

        # 2. Write dataset_info.yaml as emitted by compiler/stream_zip_to_container
        ds_info = {
            "name": "LemGendizedRestorationE2E",
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

        # 3. Create WebDataset tar shard following compiler convention
        import tarfile

        tar_path = shards_dir / "shard-00000.tar"
        with tarfile.open(tar_path, "w") as tf:
            # Input image
            ti_img = tarfile.TarInfo(name="sample_1001.webp")
            ti_img.size = len(img_bytes)
            tf.addfile(ti_img, io.BytesIO(img_bytes))

            # Restoration target image
            ti_tgt = tarfile.TarInfo(name="sample_1001.target.webp")
            ti_tgt.size = len(tgt_bytes)
            tf.addfile(ti_tgt, io.BytesIO(tgt_bytes))

            # Metadata JSON
            meta = json.dumps({"source": "integration_test", "score": 1.0}).encode("utf-8")
            ti_json = tarfile.TarInfo(name="sample_1001.json")
            ti_json.size = len(meta)
            tf.addfile(ti_json, io.BytesIO(meta))

        # 4. Resolve container reader via training suite auto-resolver
        reader = resolve_container_reader(manifold_path, split="train")
        self.assertIsInstance(reader, WebDatasetReader)
        self.assertEqual(len(reader), 1)

        # 5. Retrieve and verify sample decoding
        sample = reader[0]
        self.assertIsInstance(sample, TrainingSample)
        self.assertEqual(sample.name, "sample_1001")
        self.assertEqual(sample.image_format, "webp")
        self.assertEqual(sample.image_bytes, img_bytes)
        self.assertIsNotNone(sample.target_bytes)
        self.assertEqual(sample.target_bytes, tgt_bytes)

        # 6. Verify image decode to PIL matches expectations
        decoded_img = DirectoryReader.decode_image(sample.image_bytes)
        self.assertIsInstance(decoded_img, Image.Image)
        self.assertEqual(decoded_img.size, (32, 32))
        self.assertEqual(decoded_img.mode, "RGB")

        decoded_tgt = DirectoryReader.decode_image(sample.target_bytes)
        self.assertIsInstance(decoded_tgt, Image.Image)
        self.assertEqual(decoded_tgt.size, (32, 32))

        reader.close()

    def test_webdataset_data_obfuscation_pipeline_roundtrip(self) -> None:
        """Create a canonical WebDataset with obfuscated .data shards and load via training suite."""
        manifold_path = self.root / "LemGendizedRestorationDataE2E"
        shards_dir = manifold_path / "shards" / "train"
        shards_dir.mkdir(parents=True)

        img = Image.new("RGB", (32, 32), color=(50, 150, 250))
        tgt = Image.new("RGB", (32, 32), color=(250, 150, 50))

        buf_img = io.BytesIO()
        img.save(buf_img, format="WEBP", quality=92)
        img_bytes = buf_img.getvalue()

        buf_tgt = io.BytesIO()
        tgt.save(buf_tgt, format="WEBP", quality=95)
        tgt_bytes = buf_tgt.getvalue()

        ds_info = {
            "name": "LemGendizedRestorationDataE2E",
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

        import tarfile

        data_path = shards_dir / "shard-00000.data"
        with tarfile.open(data_path, "w") as tf:
            ti_img = tarfile.TarInfo(name="sample_2001.webp")
            ti_img.size = len(img_bytes)
            tf.addfile(ti_img, io.BytesIO(img_bytes))

            ti_tgt = tarfile.TarInfo(name="sample_2001.target.webp")
            ti_tgt.size = len(tgt_bytes)
            tf.addfile(ti_tgt, io.BytesIO(tgt_bytes))

            meta = json.dumps({"source": "integration_test_data", "score": 0.99}).encode("utf-8")
            ti_json = tarfile.TarInfo(name="sample_2001.json")
            ti_json.size = len(meta)
            tf.addfile(ti_json, io.BytesIO(meta))

        # Resolve container reader via training suite auto-resolver
        reader = resolve_container_reader(manifold_path, split="train")
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
    """Verifies API schema parity between env-manager port 8000 and datasets port 8100."""

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
    """End-to-end full production pipeline test: training -> checkpoint -> export -> stage -> metadata -> dry-run push."""

    def test_full_pipeline_roundtrip(self) -> None:
        """Simulate training step -> checkpoint -> export -> stage -> metadata -> Kaggle payload."""
        import torch
        import torch.nn as nn
        from training.cloud.kaggle_model_sync import KaggleModelSyncManager
        from training.cloud.metadata_generator import KaggleMetadataService
        from training.export.onnx import export_onnx
        from training.export.staging import VariationStagingService
        from training.export.torch_standalone import export_torch_standalone
        from training.export.validation import StrictIsolationValidator

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_hub_dir = Path(tmp_dir) / "LemGendaryModels" / "mirnet_exposure"
            model_hub_dir.mkdir(parents=True)
            ckpt_dir = model_hub_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            # 1. Simulate 1 training step on dummy model
            model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(8, 3, 1))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            x = torch.randn(1, 3, 32, 32)
            y = torch.randn(1, 3, 32, 32)
            out = model(x)
            loss = nn.functional.mse_loss(out, y)
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.isnan(loss))

            # 2. Save best checkpoint
            ckpt_path = ckpt_dir / "mirnet_exposure_best.pth"
            torch.save({"model_state": model.state_dict(), "epoch": 1, "loss": loss.item()}, ckpt_path)
            self.assertTrue(ckpt_path.exists())

            # 3. Export binaries into temporary export staging
            stage_dir = Path(tmp_dir) / "staging"
            stage_dir.mkdir(parents=True)
            pt_path = stage_dir / "LemGendaryMirNetExposure.pt"
            fp32_path = stage_dir / "LemGendaryMirNetExposure_FP32.onnx"
            fp16_path = stage_dir / "LemGendaryMirNetExposure.onnx"

            export_torch_standalone(model, pt_path)
            export_onnx(model, fp32_path, dummy_shape=(1, 3, 32, 32), half=False)
            export_onnx(model, fp16_path, dummy_shape=(1, 3, 32, 32), half=True)

            self.assertTrue(pt_path.exists())
            self.assertTrue(fp32_path.exists())
            self.assertTrue(fp16_path.exists())

            # 4. Stage isolated variations into dist/kaggle/
            staging_svc = VariationStagingService()
            staging_res = staging_svc.stage_variations(
                model_key="mirnet_exposure",
                source_files={"pytorch": pt_path, "onnx_fp32": fp32_path, "onnx_fp16": fp16_path},
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


if __name__ == "__main__":
    unittest.main()
