"""Canonical CLI entrypoint for LemGendary Model Training Suite."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.cli.lemtrain import app

if __name__ == "__main__":
    app()
