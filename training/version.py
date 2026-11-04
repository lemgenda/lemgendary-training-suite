"""Single source of the suite version: ``[project] version`` in ``pyproject.toml``.

The CLI, the sidecar app, the health route, ``openapi.json`` (generated from the app) and
the tests read :data:`__version__`. A source checkout reads ``pyproject.toml`` directly, so
a stale installed distribution (``pip install -e`` metadata) cannot win. An installed wheel
without the file falls back to the distribution metadata.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

DISTRIBUTION_NAME = "lemgendary-training-suite"
"""``[project] name`` in ``pyproject.toml``."""

PYPROJECT_FILE = Path(__file__).resolve().parent.parent / "pyproject.toml"
"""``pyproject.toml`` of the source checkout this package lives in."""

_TABLE_HEADER = re.compile(r"^\[(?P<table>[^\]]+)\]\s*$")
_VERSION_KEY = re.compile(r"^version\s*=\s*\"(?P<version>[^\"]+)\"\s*$")


def read_pyproject_version(pyproject: Path) -> str | None:
    """Return ``[project] version`` of ``pyproject``; None when the file or key is missing."""
    if not pyproject.is_file():
        return None
    table = ""
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        header = _TABLE_HEADER.match(line.strip())
        if header:
            table = header.group("table").strip()
            continue
        key = _VERSION_KEY.match(line.strip())
        if table == "project" and key:
            return key.group("version")
    return None


def resolve_version() -> str:
    """Version from ``pyproject.toml``, else from the installed distribution metadata."""
    return read_pyproject_version(PYPROJECT_FILE) or metadata.version(DISTRIBUTION_NAME)


__version__: str = resolve_version()
