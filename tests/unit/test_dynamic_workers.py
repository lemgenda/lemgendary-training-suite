"""Unit tests for dynamic resource-based DataLoader worker sizing."""

import unittest
from unittest.mock import patch

import torch

from training.data.workers import (
    _available_ram_gb,
    _default_train_workers,
    _default_val_workers,
    _system_ram_gb,
    compute_worker_topology,
)


class TestDynamicWorkers(unittest.TestCase):
    """Test dynamic resource-based worker topology calculation."""

    def test_ram_helpers(self) -> None:
        """Verify RAM helper functions return positive floats."""
        total_ram = _system_ram_gb()
        avail_ram = _available_ram_gb()
        self.assertIsInstance(total_ram, float)
        self.assertIsInstance(avail_ram, float)
        self.assertGreater(total_ram, 0.0)
        self.assertGreater(avail_ram, 0.0)

    def test_default_train_workers_low_ram(self) -> None:
        """Verify train workers scale down to 0 when available RAM is constrained."""
        with patch("sys.platform", "win32"):
            workers = _default_train_workers(
                env="local",
                ram_gb=4.0,
                available_ram_gb=1.0,
                cpu_count=8,
                cfg={},
            )
            self.assertEqual(workers, 0)

    def test_default_train_workers_ample_resources(self) -> None:
        """Verify train workers dynamically scale when cores and memory are available."""
        with patch("sys.platform", "win32"):
            workers = _default_train_workers(
                env="local",
                ram_gb=32.0,
                available_ram_gb=16.0,
                cpu_count=12,
                cfg={},
            )
            # Should scale up to maximum allowed dynamic ceiling (6 on Windows)
            self.assertGreaterEqual(workers, 4)
            self.assertLessEqual(workers, 6)

    def test_default_train_workers_respects_user_config(self) -> None:
        """Verify explicit user configuration in cfg overrides auto sizing."""
        workers = _default_train_workers(
            env="local",
            ram_gb=32.0,
            available_ram_gb=16.0,
            cpu_count=16,
            cfg={"hardware": {"num_workers": 3}},
        )
        self.assertEqual(workers, 3)

    def test_forex_domain_preserves_zero_workers(self) -> None:
        """Verify Forex models strictly enforce 0 workers to prevent Pathology 24."""
        topo = compute_worker_topology(
            env="local",
            device=torch.device("cpu"),
            is_forex=True,
            user_workers=None,
        )
        self.assertEqual(topo.num_workers, 0)
        self.assertEqual(topo.val_num_workers, 0)
        self.assertFalse(topo.persistent_workers)

    def test_dynamic_validation_workers(self) -> None:
        """Verify validation workers dynamically scale proportionally to train workers."""
        self.assertEqual(_default_val_workers("local", num_workers=0, cfg={}), 0)
        self.assertEqual(_default_val_workers("local", num_workers=2, cfg={}), 1)
        self.assertEqual(_default_val_workers("local", num_workers=6, cfg={}), 2)

    def test_persistent_workers_dynamic_activation(self) -> None:
        """Verify persistent workers activate dynamically when RAM headroom is sufficient."""
        with (
            patch("training.data.workers._system_ram_gb", return_value=32.0),
            patch("training.data.workers._available_ram_gb", return_value=12.0),
            patch("training.data.workers._default_train_workers", return_value=4),
        ):
            topo = compute_worker_topology(
                env="local",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                is_forex=False,
            )
            self.assertTrue(topo.persistent_workers)
            self.assertGreaterEqual(topo.num_workers, 4)


if __name__ == "__main__":
    unittest.main()
