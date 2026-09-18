"""Unit tests for training/utils subsystem in LemGendary Model Training Suite."""

import io
import os
import sys
import unittest
from pathlib import Path

from training.utils.paths import get_project_root, get_workspace_root, resolve_path, ensure_dir
from training.utils.logging import ForceTTY, setup_logging
from training.utils.subprocess import run_command, SubprocessExecutionError
from training.utils.interrupt import silent_worker_excepthook, register_emergency_sync
from training.utils.env_delegate import EnvManagerDelegate
from training.utils.dataset_delegate import DatasetCompilerDelegate


class TestUtilsPaths(unittest.TestCase):
    """Tests path resolution and workspace discovery."""

    def test_get_project_root(self) -> None:
        root = get_project_root()
        self.assertTrue(root.exists())
        self.assertTrue((root / "unified_models_v2.yaml").exists() or (root / "config.yaml").exists())

    def test_get_workspace_root(self) -> None:
        ws_root = get_workspace_root()
        self.assertTrue(ws_root.exists())
        self.assertTrue((ws_root / "lemgendary-training-suite").exists())

    def test_resolve_path(self) -> None:
        resolved = resolve_path("config.yaml")
        self.assertEqual(resolved, (get_project_root() / "config.yaml").resolve())

    def test_ensure_dir(self) -> None:
        tmp_dir = get_project_root() / "tmp" / "test_dir_creation"
        created = ensure_dir(tmp_dir)
        self.assertTrue(created.exists())
        self.assertTrue(created.is_dir())
        try:
            created.rmdir()
        except OSError:
            pass


class TestUtilsLogging(unittest.TestCase):
    """Tests ForceTTY and setup_logging."""

    def test_force_tty(self) -> None:
        buf = io.StringIO()
        tty = ForceTTY(buf)
        self.assertTrue(tty.isatty())
        tty.write("hello")
        tty.flush()
        self.assertEqual(buf.getvalue(), "hello")

    def test_setup_logging(self) -> None:
        logger = setup_logging(verbose=True, logger_name="test_logger")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger")


class TestUtilsSubprocess(unittest.TestCase):
    """Tests run_command wrapper."""

    def test_run_command_success(self) -> None:
        res = run_command([sys.executable, "-c", "print('subprocess_ok')"], timeout=10.0, check=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("subprocess_ok", res.stdout)

    def test_run_command_failure(self) -> None:
        with self.assertRaises(SubprocessExecutionError):
            run_command([sys.executable, "-c", "import sys; sys.exit(42)"], timeout=10.0, check=True)

    def test_run_command_timeout(self) -> None:
        with self.assertRaises(SubprocessExecutionError):
            run_command([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2, check=False)


class TestUtilsInterrupt(unittest.TestCase):
    """Tests signal and emergency handlers."""

    def test_silent_worker_excepthook(self) -> None:
        # Should return silently without raising for KeyboardInterrupt
        silent_worker_excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    def test_register_emergency_sync(self) -> None:
        called = False
        def handler() -> None:
            nonlocal called
            called = True
        register_emergency_sync(handler)


class TestUtilsDelegates(unittest.TestCase):
    """Tests EnvManagerDelegate and DatasetCompilerDelegate."""

    def test_env_delegate_instantiation(self) -> None:
        delegate = EnvManagerDelegate()
        self.assertEqual(delegate.project_root, get_project_root())
        # Online probe should return a boolean without throwing
        status = delegate.is_sidecar_online()
        self.assertIsInstance(status, bool)

    def test_dataset_delegate_instantiation(self) -> None:
        delegate = DatasetCompilerDelegate()
        self.assertEqual(delegate.workspace_root, get_workspace_root())
        status = delegate.is_sidecar_online()
        self.assertIsInstance(status, bool)
        metadata = delegate.resolve_models_metadata()
        self.assertIsInstance(metadata, dict)


if __name__ == "__main__":
    unittest.main()
