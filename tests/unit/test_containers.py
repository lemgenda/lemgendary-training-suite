"""Unit tests for LemGendary modern container reader plugins."""

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from training.data.containers import (
    DirectoryReader,
    LitDataReader,
    MdsReader,
    ParquetReader,
    Sample,
    WebDatasetReader,
    resolve_container_reader,
)


class TestContainers(unittest.TestCase):
    """Test suite covering modern container readers and resolution factory."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_directory_reader_lossless_and_legacy(self) -> None:
        manifold = self.root / "test_dir_manifold"
        images_dir = manifold / "images" / "train"
        targets_dir = manifold / "targets" / "train"
        masks_dir = manifold / "masks" / "train"
        images_dir.mkdir(parents=True)
        targets_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create sample 1 (WebP)
        img1 = Image.new("RGB", (32, 32), color=(255, 0, 0))
        img1.save(images_dir / "sample_001.webp", format="WEBP", lossless=True)
        img1.save(targets_dir / "sample_001.webp", format="WEBP", lossless=True)

        # Create sample 2 (PNG) with mask
        img2 = Image.new("RGB", (32, 32), color=(0, 255, 0))
        img2.save(images_dir / "sample_002.png", format="PNG")
        mask2 = Image.new("L", (32, 32), color=128)
        mask2.save(masks_dir / "sample_002.png", format="PNG")

        reader = DirectoryReader(manifold, split="train")
        self.assertEqual(len(reader), 2)

        sample1 = reader[0]
        self.assertIsInstance(sample1, Sample)
        self.assertEqual(sample1.name, "sample_001")
        self.assertEqual(sample1.image_format, "webp")
        self.assertIsNotNone(sample1.target_bytes)

        # Verify decode
        decoded1 = DirectoryReader.decode_image(sample1.image_bytes)
        self.assertEqual(decoded1.size, (32, 32))

        sample2 = reader[1]
        self.assertEqual(sample2.name, "sample_002")
        self.assertEqual(sample2.image_format, "png")
        self.assertIsNotNone(sample2.mask_bytes)

        reader.close()

    def test_parquet_reader(self) -> None:
        manifold = self.root / "test_parquet_manifold"
        manifold.mkdir(parents=True)

        # Create dummy image bytes
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), color=(10, 20, 30)).save(buf, format="WEBP")
        raw_bytes = buf.getvalue()

        table = pa.Table.from_arrays(
            [
                pa.array(["sample_101", "sample_102"]),
                pa.array([raw_bytes, raw_bytes], type=pa.binary()),
                pa.array([1, 0]),
            ],
            names=["name", "image", "label"],
        )
        pq.write_table(table, manifold / "train.parquet", row_group_size=1)

        reader = ParquetReader(manifold, split="train")
        self.assertEqual(len(reader), 2)

        sample = reader[0]
        self.assertEqual(sample.name, "sample_101")
        self.assertEqual(len(sample.image_bytes), len(raw_bytes))
        self.assertEqual(sample.label, 1)

        reader.close()

    def test_webdataset_reader(self) -> None:
        manifold = self.root / "test_wds_manifold"
        shards_dir = manifold / "shards"
        shards_dir.mkdir(parents=True)
        tar_path = shards_dir / "shard-000000.tar"

        buf_img = io.BytesIO()
        Image.new("RGB", (16, 16), color=(50, 60, 70)).save(buf_img, format="WEBP")
        img_bytes = buf_img.getvalue()

        buf_tgt = io.BytesIO()
        Image.new("RGB", (16, 16), color=(255, 255, 255)).save(buf_tgt, format="WEBP")
        tgt_bytes = buf_tgt.getvalue()

        buf_mask = io.BytesIO()
        Image.new("L", (16, 16), color=128).save(buf_mask, format="WEBP")
        mask_bytes = buf_mask.getvalue()

        meta_json = json.dumps({
            "distribution": [0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05],
            "task": "restoration",
        }).encode("utf-8")

        with tarfile.open(tar_path, mode="w") as tf:
            # Add input image
            ti_img = tarfile.TarInfo(name="item_001.webp")
            ti_img.size = len(img_bytes)
            tf.addfile(ti_img, io.BytesIO(img_bytes))

            # Add paired target
            ti_tgt = tarfile.TarInfo(name="item_001.target.webp")
            ti_tgt.size = len(tgt_bytes)
            tf.addfile(ti_tgt, io.BytesIO(tgt_bytes))

            # Add paired mask
            ti_msk = tarfile.TarInfo(name="item_001.mask.webp")
            ti_msk.size = len(mask_bytes)
            tf.addfile(ti_msk, io.BytesIO(mask_bytes))

            # Add JSON metadata with distribution
            ti_json = tarfile.TarInfo(name="item_001.json")
            ti_json.size = len(meta_json)
            tf.addfile(ti_json, io.BytesIO(meta_json))

        reader = WebDatasetReader(manifold, split="train")
        self.assertEqual(len(reader), 1)

        sample = reader[0]
        self.assertEqual(sample.name, "item_001")
        self.assertEqual(sample.image_format, "webp")
        self.assertEqual(sample.image_bytes, img_bytes)
        self.assertEqual(sample.target_bytes, tgt_bytes)
        self.assertEqual(sample.mask_bytes, mask_bytes)
        self.assertIsInstance(sample.label, list)
        self.assertEqual(len(sample.label), 10)
        self.assertEqual(sample.metadata.get("json", {}).get("task"), "restoration")

        reader.close()

    def test_webdataset_extracted_shards_reader(self) -> None:
        """Test WebDatasetReader discovering unpacked shard directories (Kaggle layout)."""
        manifold = self.root / "test_wds_extracted"
        shard_dir = manifold / "shards" / "train" / "shard-00000"
        shard_dir.mkdir(parents=True)

        buf_img = io.BytesIO()
        Image.new("RGB", (16, 16), color=(200, 100, 50)).save(buf_img, format="WEBP")
        img_bytes = buf_img.getvalue()

        buf_tgt = io.BytesIO()
        Image.new("RGB", (16, 16), color=(50, 100, 200)).save(buf_tgt, format="WEBP")
        tgt_bytes = buf_tgt.getvalue()

        meta_json = json.dumps({"task": "restoration", "distribution": [0.1] * 10}).encode("utf-8")

        (shard_dir / "sample_001.webp").write_bytes(img_bytes)
        (shard_dir / "sample_001.target.webp").write_bytes(tgt_bytes)
        (shard_dir / "sample_001.json").write_bytes(meta_json)

        reader = WebDatasetReader(manifold, split="train")
        self.assertEqual(len(reader), 1)

        sample = reader[0]
        self.assertEqual(sample.name, "sample_001")
        self.assertEqual(sample.image_format, "webp")
        self.assertEqual(sample.image_bytes, img_bytes)
        self.assertEqual(sample.target_bytes, tgt_bytes)
        self.assertIsInstance(sample.label, list)
        self.assertEqual(len(sample.label), 10)
        self.assertEqual(sample.metadata.get("json", {}).get("task"), "restoration")

        reader.close()

    def test_mds_reader(self) -> None:
        manifold = self.root / "test_mds_manifold"
        mds_dir = manifold / "mds"
        mds_dir.mkdir(parents=True)

        index_content = {
            "version": 2,
            "shards": [
                {"samples": 5, "raw_data": {"basename": "shard.00000.mds"}},
                {"samples": 5, "raw_data": {"basename": "shard.00001.mds"}},
            ],
        }
        (mds_dir / "index.json").write_text(json.dumps(index_content), encoding="utf-8")

        reader = MdsReader(manifold, split="train")
        self.assertEqual(len(reader), 10)

        sample = reader[3]
        self.assertEqual(sample.name, "mds_sample_00000003")
        self.assertEqual(sample.metadata.get("shard"), "shard.00000.mds")

        sample_second_shard = reader[7]
        self.assertEqual(sample_second_shard.metadata.get("shard"), "shard.00001.mds")

        reader.close()

    def test_litdata_reader(self) -> None:
        manifold = self.root / "test_lit_manifold"
        lit_dir = manifold / "litdata"
        lit_dir.mkdir(parents=True)
        (lit_dir / "chunk-0.bin").touch()
        (lit_dir / "chunk-1.bin").touch()

        reader = LitDataReader(manifold, split="train")
        self.assertEqual(len(reader), 2)

        sample = reader[1]
        self.assertEqual(sample.name, "lit_sample_00000001")
        reader.close()

    def test_resolve_container_reader(self) -> None:
        manifold = self.root / "test_resolution"
        images_dir = manifold / "images" / "train"
        images_dir.mkdir(parents=True)
        (images_dir / "sample.jpg").touch()

        # 1. Automatic DirectoryReader detection
        reader = resolve_container_reader(manifold, split="train")
        self.assertIsInstance(reader, DirectoryReader)
        reader.close()

        # 2. Modern dataset_info.yaml format declarations
        # format: webdataset
        info_data_wds = {"format": "webdataset"}
        (manifold / "dataset_info.yaml").write_text(yaml.dump(info_data_wds), encoding="utf-8")
        shards_dir = manifold / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(shards_dir / "shard-00000.tar", mode="w") as tf:
            ti = tarfile.TarInfo(name="item_000.webp")
            ti.size = 0
            tf.addfile(ti, io.BytesIO(b""))

        wds_reader = resolve_container_reader(manifold, split="train")
        self.assertIsInstance(wds_reader, WebDatasetReader)
        wds_reader.close()

        # canonical_format: litdata
        info_data_lit = {"canonical_format": "litdata"}
        (manifold / "dataset_info.yaml").write_text(yaml.dump(info_data_lit), encoding="utf-8")
        lit_dir = manifold / "litdata"
        lit_dir.mkdir(parents=True, exist_ok=True)
        (lit_dir / "chunk-0.bin").touch()

        lit_reader = resolve_container_reader(manifold, split="train")
        self.assertIsInstance(lit_reader, LitDataReader)
        lit_reader.close()

        # format: parquet
        info_data = {"format": "parquet"}
        (manifold / "dataset_info.yaml").write_text(yaml.dump(info_data), encoding="utf-8")
        table = pa.Table.from_arrays([pa.array(["s1"])], names=["id"])
        pq.write_table(table, manifold / "data.parquet")

        pq_reader = resolve_container_reader(manifold, split="train")
        self.assertIsInstance(pq_reader, ParquetReader)
        pq_reader.close()


if __name__ == "__main__":
    unittest.main()
