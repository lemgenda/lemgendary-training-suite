"""The suite version has one source: ``[project] version`` in ``pyproject.toml``.

Everything else derives it (``training.version.__version__``) or is checked against it here:
the README status line, the generated ``openapi.json`` and the console-script target. These
tests are torch-free.
"""

from __future__ import annotations

import importlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import yaml

from training.version import PYPROJECT_FILE, __version__, read_pyproject_version

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCANNED_SOURCE_DIRS = ("training", "tools", "models", "data", "export")


class TestVersionSource(unittest.TestCase):
    """``__version__`` comes from ``pyproject.toml`` and nothing else hardcodes it."""

    def test_version_matches_pyproject(self) -> None:
        """``__version__`` is the ``[project] version`` of the checkout's pyproject.toml."""
        self.assertEqual(PYPROJECT_FILE, PROJECT_ROOT / "pyproject.toml")
        self.assertEqual(__version__, read_pyproject_version(PYPROJECT_FILE))
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")

    def test_reader_ignores_versions_of_other_tables(self) -> None:
        """Only the ``[project]`` table's ``version`` key counts."""
        text = (
            '[tool.other]\nversion = "0.0.1"\n\n'
            '[project]\nname = "x"\nauthors = [{name = "a"}]\nversion = "1.2.3"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pyproject.toml"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(read_pyproject_version(path), "1.2.3")
            self.assertIsNone(read_pyproject_version(Path(tmp) / "missing.toml"))

    def test_no_hardcoded_version_in_sources(self) -> None:
        """Python sources and config.yaml do not repeat the version literal."""
        literal = re.compile(rf"\b{re.escape(__version__)}\b")
        offenders = [
            str(path.relative_to(PROJECT_ROOT))
            for folder in SCANNED_SOURCE_DIRS
            for path in sorted((PROJECT_ROOT / folder).rglob("*.py"))
            if literal.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [])
        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("version", config)


class TestVersionConsumers(unittest.TestCase):
    """Files that must show the version agree with ``pyproject.toml``."""

    def test_readme_status_line(self) -> None:
        """README ``Current Status`` names the pyproject version."""
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(r"^- \*\*Version\*\*: `v(?P<version>[^`-]+)", readme, re.MULTILINE)
        if match is None:
            self.fail("README.md has no '- **Version**: `v...`' status line")
        self.assertEqual(match.group("version"), __version__)

    def test_openapi_info_version(self) -> None:
        """The committed ``openapi.json`` carries the pyproject version."""
        spec = json.loads((PROJECT_ROOT / "openapi.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["info"]["version"], __version__)

    def test_console_script_target_resolves(self) -> None:
        """``[project.scripts] lemtrain`` names an importable, packaged callable."""
        text = PYPROJECT_FILE.read_text(encoding="utf-8")
        match = re.search(r'^lemtrain\s*=\s*"(?P<module>[\w.]+):(?P<attr>\w+)"', text, re.MULTILINE)
        if match is None:
            self.fail("pyproject.toml declares no lemtrain console script")
        self.assertTrue(match.group("module").startswith("training."))
        module = importlib.import_module(match.group("module"))
        self.assertTrue(callable(getattr(module, match.group("attr"))))


if __name__ == "__main__":
    unittest.main()
