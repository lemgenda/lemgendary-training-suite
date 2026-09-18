"""Unit tests for cloud subsystem: credentials, git, kaggle, and gdrive bit-exact parity."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from training.cloud.credentials import (
    mask_secret,
    resolve_gdrive_credentials,
    resolve_github_credentials,
    resolve_kaggle_credentials,
)
from training.cloud.gdrive import GDriveSyncManager, compute_file_sha256
from training.cloud.git_hub import GitHubSyncManager
from training.cloud.kaggle_hub import KaggleHubManager
from training.cloud.manager import CloudManager, CloudSyncError


class TestCloudCredentials(unittest.TestCase):
    """Tests for credentials parsing and stealth masking."""

    def test_mask_secret(self) -> None:
        raw = "git clone https://ghp_secretToken12345@github.com/repo.git"
        secret = "ghp_secretToken12345"
        masked = mask_secret(raw, secret)
        self.assertNotIn(secret, masked)
        self.assertIn("***STEALTH***", masked)

        # None or short secret should not break
        self.assertEqual(mask_secret("hello world", None), "hello world")
        self.assertEqual(mask_secret("hello world", "ab"), "hello world")

    def test_resolve_credentials_overrides(self) -> None:
        gh_token = resolve_github_credentials(override_pat="test_gh_token")
        self.assertEqual(gh_token, "test_gh_token")

        k_user, k_key = resolve_kaggle_credentials(override_user="custom_user", override_key="custom_key")
        self.assertEqual(k_user, "custom_user")
        self.assertEqual(k_key, "custom_key")

        gd_token = resolve_gdrive_credentials(override_token="test_gd_token")
        self.assertEqual(gd_token, "test_gd_token")


class TestCloudSyncError(unittest.TestCase):
    """Tests for structured CloudSyncError."""

    def test_error_attributes(self) -> None:
        err = CloudSyncError(code="NET_ERR", message="Connection reset", retryable=True)
        self.assertEqual(err.code, "NET_ERR")
        self.assertEqual(err.message, "Connection reset")
        self.assertTrue(err.retryable)
        self.assertIn("[NET_ERR]", str(err))
        self.assertIn("retryable=True", str(err))


class TestGitHubSyncManager(unittest.TestCase):
    """Tests for GitHubSyncManager setup and health probing."""

    def test_probe_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = GitHubSyncManager(hub_root=Path(tmp), hub_user="testuser")
            health = mgr.probe_health()
            self.assertEqual(health["provider"], "github")
            self.assertTrue(health["hub_root_exists"])
            self.assertEqual(health["hub_repo"], "testuser/lemgendary-pretrained-models")


class TestKaggleHubManager(unittest.TestCase):
    """Tests for KaggleHubManager handle and bundle creation."""

    def test_handle_generation(self) -> None:
        mgr = KaggleHubManager(username="tester")
        handle = mgr.get_handle("mirnet_exposure")
        self.assertEqual(handle, "tester/lemgendary-mirnet-exposure-checkpoints/pytorch/default")

        # NIMA aesthetic normalization
        nima_handle = mgr.get_handle("nima_aesthetic")
        self.assertEqual(nima_handle, "tester/lemgendary-nima-aesthetics-checkpoints/pytorch/default")

    def test_create_cloud_kernel_bundle(self) -> None:
        mgr = KaggleHubManager(username="tester")
        bundle_dir = mgr.create_cloud_kernel_bundle("test_model", gpu="T4")
        try:
            self.assertTrue(bundle_dir.exists())
            self.assertTrue((bundle_dir / "kernel-metadata.json").exists())
            self.assertTrue((bundle_dir / "train_cloud.py").exists())
        finally:
            import shutil
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir, ignore_errors=True)


class TestGDriveBitExactParity(unittest.TestCase):
    """Gate 7: Verify bit-exact SHA256 parity during push and pull."""

    def test_push_and_pull_bit_exact_parity(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as mount_tmp, tempfile.TemporaryDirectory() as dest_tmp:
            source_dir = Path(source_tmp)
            mount_dir = Path(mount_tmp)
            dest_dir = Path(dest_tmp)

            # 1. Create a deterministic fixture model checkpoint
            ckpt_dir = source_dir / "checkpoints"
            ckpt_dir.mkdir()
            payload = b"LEMGENDARY_TEST_CHECKPOINT_DATA_1234567890_XYZ"
            ckpt_file = ckpt_dir / "best_model.pth"
            ckpt_file.write_bytes(payload)

            src_hash = compute_file_sha256(ckpt_file)

            # 2. Sync to simulated Google Drive mount
            mgr = GDriveSyncManager(mount_root=mount_dir)
            sync_ok = mgr.sync("test_model", epoch=1, src_dir=source_dir)
            self.assertTrue(sync_ok)

            # Verify synced file on mount
            mounted_ckpt = mount_dir / "LemGendaryModels" / "test_model" / "checkpoints" / "best_model.pth"
            self.assertTrue(mounted_ckpt.exists())
            self.assertEqual(compute_file_sha256(mounted_ckpt), src_hash)

            # 3. Pull back to a new destination directory
            pull_ok = mgr.pull("test_model", dest_dir=dest_dir)
            self.assertTrue(pull_ok)

            # 4. Verify bit-exact SHA256 digest on pulled artifact
            pulled_ckpt = dest_dir / "checkpoints" / "best_model.pth"
            self.assertTrue(pulled_ckpt.exists())
            pulled_hash = compute_file_sha256(pulled_ckpt)

            self.assertEqual(pulled_hash, src_hash)


if __name__ == "__main__":
    unittest.main()
