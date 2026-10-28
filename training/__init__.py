"""LemGendary Training Suite package initialization."""

import os
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

