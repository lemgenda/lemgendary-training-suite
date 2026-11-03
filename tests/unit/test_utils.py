"""Unit tests for training/utils subsystem in LemGendary Model Training Suite."""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.utils.dataset_delegate import DatasetCompilerDelegate
from training.utils.env_delegate import EnvManagerDelegate
from training.utils.interrupt import register_emergency_sync, silent_worker_excepthook
from training.utils.logging import ForceTTY, setup_logging
from training.utils.paths import (
    ensure_dir,
    get_project_root,
    get_workspace_root,
    resolve_path,
)
from training.utils.subprocess import SubprocessExecutionError, run_command


class TestUtilsPaths(unittest.TestCase):
    """Tests path resolution and workspace discovery."""

    def test_get_project_root(self) -> None:
        """Verify get_project_root points at the training-suite checkout."""
        root = get_project_root()
        self.assertTrue(root.exists())
        self.assertTrue(
            (root / "unified_models_v2.yaml").exists() or (root / "config.yaml").exists()
        )

    def test_get_workspace_root(self) -> None:
        """Verify get_workspace_root honours the env override, else is the project's parent."""
        fake = Path(os.environ["LEMGENDARY_WORKSPACE_ROOT"]).resolve()
        self.assertEqual(get_workspace_root(), fake)
        with mock.patch.dict(os.environ):
            del os.environ["LEMGENDARY_WORKSPACE_ROOT"]
            ws_root = get_workspace_root()
        self.assertEqual(ws_root, get_project_root().parent)
        self.assertTrue((ws_root / get_project_root().name).exists())

    def test_resolve_path(self) -> None:
        """Verify resolve_path resolves relative to the project root."""
        resolved = resolve_path("config.yaml")
        self.assertEqual(resolved, (get_project_root() / "config.yaml").resolve())

    def test_ensure_dir(self) -> None:
        """Verify ensure_dir creates the directory."""
        with tempfile.TemporaryDirectory() as base:
            created = ensure_dir(Path(base) / "nested" / "test_dir_creation")
            self.assertTrue(created.exists())
            self.assertTrue(created.is_dir())


class TestUtilsLogging(unittest.TestCase):
    """Tests ForceTTY and setup_logging."""

    def test_force_tty(self) -> None:
        """Verify ForceTTY reports a TTY and writes through."""
        buf = io.StringIO()
        tty = ForceTTY(buf)
        self.assertTrue(tty.isatty())
        tty.write("hello")
        tty.flush()
        self.assertEqual(buf.getvalue(), "hello")

    def test_setup_logging(self) -> None:
        """Verify setup_logging returns the named logger."""
        logger = setup_logging(verbose=True, logger_name="test_logger")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger")


class TestUtilsSubprocess(unittest.TestCase):
    """Tests run_command wrapper."""

    def test_run_command_success(self) -> None:
        """Verify run_command captures output on success."""
        res = run_command(
            [sys.executable, "-c", "print('subprocess_ok')"], timeout=10.0, check=True
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("subprocess_ok", res.stdout)

    def test_run_command_failure(self) -> None:
        """Verify run_command raises on a non-zero exit with check=True."""
        with self.assertRaises(SubprocessExecutionError):
            run_command(
                [sys.executable, "-c", "import sys; sys.exit(42)"], timeout=10.0, check=True
            )

    def test_run_command_timeout(self) -> None:
        """Verify run_command raises on timeout."""
        with self.assertRaises(SubprocessExecutionError):
            run_command(
                [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2, check=False
            )


class TestUtilsInterrupt(unittest.TestCase):
    """Tests signal and emergency handlers."""

    def test_silent_worker_excepthook(self) -> None:
        """Verify the silent worker excepthook swallows KeyboardInterrupt."""
        # Should return silently without raising for KeyboardInterrupt
        silent_worker_excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    def test_register_emergency_sync(self) -> None:
        """Verify an emergency sync handler can be registered."""
        called = False

        def handler() -> None:
            nonlocal called
            called = True

        register_emergency_sync(handler)


class TestUtilsDelegates(unittest.TestCase):
    """Tests EnvManagerDelegate and DatasetCompilerDelegate."""

    def test_env_delegate_instantiation(self) -> None:
        """Verify the env-manager delegate instantiates and probes safely."""
        delegate = EnvManagerDelegate()
        self.assertEqual(delegate.project_root, get_project_root())
        # Online probe should return a boolean without throwing
        status = delegate.is_sidecar_online()
        self.assertIsInstance(status, bool)

    def test_dataset_delegate_instantiation(self) -> None:
        """Verify the dataset-compiler delegate instantiates and probes safely."""
        delegate = DatasetCompilerDelegate()
        self.assertEqual(delegate.workspace_root, get_workspace_root())
        status = delegate.is_sidecar_online()
        self.assertIsInstance(status, bool)
        metadata = delegate.resolve_models_metadata()
        self.assertIsInstance(metadata, dict)


if __name__ == "__main__":
    unittest.main()
