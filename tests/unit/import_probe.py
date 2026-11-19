"""Fresh-interpreter import probe shared by the import-hygiene tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = (
    "import importlib, json, sys\n"
    "importlib.import_module(sys.argv[1])\n"
    "watched = sys.argv[2:]\n"
    "loaded = sorted(\n"
    "    m for m in sys.modules if any(m == w or m.startswith(w + '.') for w in watched)\n"
    ")\n"
    "print(json.dumps(loaded))\n"
)


def loaded_after_import(module: str, watched: tuple[str, ...]) -> tuple[list[str], str]:
    """Import ``module`` in a fresh interpreter.

    Returns the watched modules found in ``sys.modules`` afterwards and the combined
    stdout and stderr of the probe.
    """
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE, module, *watched],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    loaded = json.loads(probe.stdout.strip().splitlines()[-1])
    return loaded, probe.stdout + probe.stderr
