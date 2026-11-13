"""The installed ``lemtrain`` console script starts outside the source checkout.

``training.cli.lemtrain`` imports ``tools.judicial_audit``; the ``tools`` package must be part
of the distribution (``[tool.setuptools.packages.find] include``) or the installed script
fails with ``No module named 'tools'`` whenever the checkout is not the working directory.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

import tomllib

_SUITE_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_DIRS = ("training", "models", "data", "export", "tools")
"""Top-level packages of the checkout that suite code imports."""


def _included_patterns() -> list[str]:
    pyproject = tomllib.loads((_SUITE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return list(pyproject["tool"]["setuptools"]["packages"]["find"]["include"])


def test_distribution_includes_every_suite_package() -> None:
    """Each top-level suite package is matched by a ``packages.find`` include pattern."""
    patterns = _included_patterns()
    for package in _PACKAGE_DIRS:
        assert (_SUITE_ROOT / package).is_dir()
        assert f"{package}*" in patterns, package


def _console_script() -> Path:
    scripts = Path(sysconfig.get_path("scripts"))
    suffix = ".exe" if sys.platform == "win32" else ""
    return scripts / f"lemtrain{suffix}"


def test_installed_console_script_runs_outside_checkout(tmp_path: Path) -> None:
    """``lemtrain --help`` exits 0 when started from an unrelated folder."""
    script = _console_script()
    assert script.is_file(), f"{script} missing: run 'pip install -e .' in the suite"
    result = subprocess.run(
        [str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "kaggle" in result.stdout
