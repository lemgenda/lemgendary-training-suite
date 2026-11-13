"""One trainer log call reaches the notebook as exactly one line.

Kaggle run 4 of ``upn_v2`` showed every trainer line twice: the Kaggle training cell let the
trainer inherit the kernel's file descriptors, and ipykernel forwards fd output both to its
terminal and as notebook output. The generated launch now pipes the trainer and prints each
line from the cell. The test runs that generated code with a child that logs one record and
checks the record arrives once through the cell's ``print`` and never on the raw fds.
"""

from __future__ import annotations

import io
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from training.notebooks.cells import sync as sync_cells
from training.notebooks.registry import load_registry, resolve_model_metadata

_KEY = "upn_v2"
_RECORD = "Image decode failed for Manifold[7]; substituting neutral gray"
_CHILD = (
    "import logging, sys\n"
    "from training.utils.logging import setup_logging\n"
    "setup_logging(logger_name='lemtrain')\n"
    f"logging.getLogger('lemtrain.dataset').warning({_RECORD!r})\n"
    f"logging.getLogger('unconfigured.child').warning({_RECORD!r})\n"
    "print('\\r  [BATCH 1/2]', end='', flush=True)\n"
    "print('\\r  [BATCH 2/2]', flush=True)\n"
)
_HANDLER = "except KeyboardInterrupt:\n    p.kill()\n    raise\n"


def _run_generated_launch(cell_path: Path, cell_out: io.StringIO) -> str:
    """Run the generated launch lines with the logging child; return the cell's output.

    ``cell_out`` stands in for the notebook output stream (``sys.stdout`` of the cell).
    """
    cell_path.write_text("".join(sync_cells.TRAINER_STREAM_LINES) + _HANDLER, encoding="utf-8")
    namespace = runpy.run_path(
        str(cell_path),
        init_globals={
            "subprocess": subprocess,
            "sys": sys,
            "cmd": [sys.executable, "-u", "-c", _CHILD],
        },
    )
    assert namespace["p"].returncode == 0
    return cell_out.getvalue()


def test_one_log_call_is_one_notebook_line(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each record is printed by the cell once; nothing goes to the inherited fds."""
    cell_out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", cell_out)
    cell_output = _run_generated_launch(tmp_path / "training_cell.py", cell_out)
    raw = capfd.readouterr()

    lines = cell_output.splitlines()
    assert sum(_RECORD in line for line in lines) == 2  # one per logger, once each
    assert len([line for line in lines if line.startswith("[")]) == 1
    assert "\r  [BATCH 1/2]\r  [BATCH 2/2]" in cell_output
    assert _RECORD not in raw.out
    assert _RECORD not in raw.err


@pytest.mark.parametrize("target", ["kaggle", "colab"])
def test_every_target_pipes_the_trainer(target: str) -> None:
    """Kaggle and Colab training cells use the same piped launch."""
    meta = resolve_model_metadata(_KEY, load_registry())
    source = "".join(sync_cells.build_training_cell(meta, target)["source"])
    assert "".join(sync_cells.TRAINER_STREAM_LINES) in source
    assert "subprocess.Popen(cmd)\n" not in source
