"""Stored epochs of checkpoint files: ``python -m training.utils.epoch_probe <path>...``.

The notebook checkpoint recovery cell runs this module in a fresh interpreter instead of
importing :func:`training.utils.model_payload.checkpoint_epoch` into the notebook's own
kernel process: that process may hold a torch module loaded before the dependency cell
installed another build. The result is one JSON object on the last stdout line, mapping each
path argument to its stored epoch (``null`` when the file has none or is unreadable).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from training.utils.model_payload import checkpoint_epoch


def stored_epochs(paths: Sequence[str]) -> dict[str, int | None]:
    """The epoch stored in each checkpoint of ``paths``, keyed by the path as given."""
    return {path: checkpoint_epoch(path) for path in paths}


def main(argv: Sequence[str] | None = None) -> int:
    """Print the stored epochs of the paths in ``argv`` as one JSON line; return 0."""
    paths = list(sys.argv[1:] if argv is None else argv)
    print(json.dumps(stored_epochs(paths)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
