"""Notebook sync into dataset manifold folders skips read-only folders quietly.

On Kaggle the manifold folders under ``LemGendaryDatasets`` link to the read-only
``/kaggle/input``; every generated notebook used to log one ``Read-only file system``
warning per file and process.
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

import pytest

from training.notebooks.builders import base
from training.utils.paths import WORKSPACE_ROOT_ENV

MODEL_KEY = "foo_model"
REGISTRY = {MODEL_KEY: {"datasets": ["LemGendizedFoo"]}}
FILES = ("foo_kaggle_training.ipynb", "foo_colab_training.ipynb")


@pytest.fixture(name="manifold_dir")
def _manifold_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    """``LemGendaryDatasets/LemGendizedFoo`` in the fake workspace; no folder reported yet."""
    monkeypatch.setattr(base, "_REPORTED_READ_ONLY", set())
    folder = Path(os.environ[WORKSPACE_ROOT_ENV]) / "LemGendaryDatasets" / "LemGendizedFoo"
    folder.mkdir(parents=True)
    return folder


def _sync_all() -> None:
    for filename in FILES:
        base.sync_manifold_notebooks(MODEL_KEY, "{}", filename, REGISTRY)


def test_writable_manifold_gets_the_notebooks(manifold_dir: Path) -> None:
    """A writable manifold folder receives every notebook."""
    _sync_all()
    assert sorted(p.name for p in manifold_dir.iterdir()) == sorted(FILES)


def test_read_only_manifold_is_reported_once(
    manifold_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A folder without write access is skipped with one info record, no warnings."""
    real_access = os.access

    def no_write(path: str | os.PathLike[str], mode: int) -> bool:
        if Path(path) == manifold_dir and mode == os.W_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(base.os, "access", no_write)
    with caplog.at_level(logging.INFO, logger=base.logger.name):
        _sync_all()
        _sync_all()
    assert not list(manifold_dir.iterdir())
    records = [r for r in caplog.records if r.name == base.logger.name]
    assert [r.levelno for r in records] == [logging.INFO]
    assert "read-only" in records[0].getMessage()


def test_read_only_file_system_error_is_reported_once(
    manifold_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A write refused with EROFS (access looked fine) is one info record per folder."""

    def read_only_open(file: object, *_args: object, **_kwargs: object):
        raise OSError(errno.EROFS, "Read-only file system", str(file))

    monkeypatch.setattr(base, "open", read_only_open, raising=False)
    with caplog.at_level(logging.INFO, logger=base.logger.name):
        _sync_all()
    records = [r for r in caplog.records if r.name == base.logger.name]
    assert [r.levelno for r in records] == [logging.INFO]
    assert not list(manifold_dir.iterdir())
