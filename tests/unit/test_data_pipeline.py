"""Unit tests for LemGendary Model Training Suite data pipeline subsystem."""

import io
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import torch
from torch.utils.data import Dataset

from training.data.degrade import (
    DynamicOnTheFlyDegrader,
    JpegCompressionGuard,
    apply_film_degradation,
    apply_synthetic_degradation,
    synthesize_degradation,
)
from training.data.loaders import build_train_loader, build_val_loader, rebuild_train_loader
from training.data.manifold import ManifoldInfo, ManifoldResolver
from training.data.workers import WorkerTopology, compute_worker_topology, dispose_loader


class _DummyDataset(Dataset):
    def __init__(self, size: int = 16, task_type: str = "restoration") -> None:
        self.size_val = size
        self.task_type = task_type
        self.size = (64, 64)

    def __len__(self) -> int:
        return self.size_val

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(3, 64, 64, dtype=torch.float32)
        y = torch.ones(3, 64, 64, dtype=torch.float32)
        return x, y


class TestDataPipeline(unittest.TestCase):
    """Test suite covering manifold resolution, worker topology, loaders, and degradations."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_manifold_candidates(self) -> None:
        resolver = ManifoldResolver(project_root=self.root_path, env="local")
        candidates = resolver.generate_name_candidates("MyManifold")
        self.assertIn("MyManifold", candidates)
        self.assertIn("LemGendizedMyManifold", candidates)

        kaggle_resolver = ManifoldResolver(project_root=self.root_path, env="kaggle")
        kaggle_cands = kaggle_resolver.generate_name_candidates("MyManifold")
        self.assertIn("MyManifoldKaggleReady", kaggle_cands)

    def test_manifold_inspection(self) -> None:
        manifold_dir = self.root_path / "LemGendizedTestManifold"
        manifold_dir.mkdir(parents=True)
        images_dir = manifold_dir / "images" / "train"
        images_dir.mkdir(parents=True)
        (images_dir / "sample_001.png").touch()

        resolver = ManifoldResolver(project_root=self.root_path, env="local")
        info = resolver.inspect_manifold(manifold_dir)

        self.assertIsInstance(info, ManifoldInfo)
        self.assertEqual(info.name, "LemGendizedTestManifold")
        self.assertEqual(info.container_type, "directory")
        self.assertEqual(info.task_type, "restoration")
        self.assertIsNotNone(info.images_dir)

        # Test split path resolution
        split_path = resolver.resolve_split_path(
            manifold_path=manifold_dir,
            folder_name="images",
            split="train",
            filename="sample_001.png",
        )
        self.assertIsNotNone(split_path)
        self.assertTrue(split_path.exists())

    def test_worker_topology_calculation(self) -> None:
        topo = compute_worker_topology(
            env="local",
            device=torch.device("cpu"),
            is_forex=False,
            user_num_workers=2,
            user_val_workers=1,
        )
        self.assertIsInstance(topo, WorkerTopology)
        self.assertEqual(topo.num_workers, 2)
        self.assertEqual(topo.val_num_workers, 1)
        self.assertFalse(topo.pin_memory)

        # Forex dataset forces 0 workers on time-series streaming
        forex_topo = compute_worker_topology(
            env="local",
            device=torch.device("cpu"),
            is_forex=True,
        )
        self.assertEqual(forex_topo.num_workers, 0)

    def test_dataloader_factories_and_lifecycle(self) -> None:
        dataset = _DummyDataset(size=8)
        train_loader = build_train_loader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        self.assertEqual(len(train_loader), 4)

        # Batch iteration
        for batch_x, batch_y in train_loader:
            self.assertEqual(batch_x.shape, (2, 3, 64, 64))
            self.assertEqual(batch_y.shape, (2, 3, 64, 64))
            break

        # Rebuild loader cleanly
        rebuilt = rebuild_train_loader(
            old_loader=train_loader,
            dataset=dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        self.assertEqual(len(rebuilt), 2)
        dispose_loader(rebuilt)

        val_loader = build_val_loader(
            dataset=dataset,
            batch_size=4,
            num_workers=0,
        )
        self.assertEqual(len(val_loader), 2)
        dispose_loader(val_loader)

    def test_dynamic_degradation(self) -> None:
        tensor = torch.ones(3, 32, 32, dtype=torch.float32) * 0.5
        degraded = apply_synthetic_degradation(tensor, deg=0.5, theta=1.0, conf=0.5)
        self.assertEqual(degraded.shape, (3, 32, 32))
        self.assertGreaterEqual(float(degraded.min()), 0.0)
        self.assertLessEqual(float(degraded.max()), 1.0)

        film_degraded = apply_film_degradation(tensor)
        self.assertEqual(film_degraded.shape, (3, 32, 32))

        degrader = DynamicOnTheFlyDegrader()
        out_tensor, meta = degrader(tensor, sample_seed=42)
        self.assertEqual(out_tensor.shape, (3, 32, 32))
        self.assertIsInstance(meta, dict)

    def test_jpeg_guard_and_pil_degradation(self) -> None:
        pil_img = Image.new("RGB", (64, 64), color=(128, 64, 32))
        guard = JpegCompressionGuard(probability=1.0)
        guarded = guard(pil_img)
        self.assertEqual(guarded.size, (64, 64))

        synth = synthesize_degradation(pil_img)
        self.assertEqual(synth.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
