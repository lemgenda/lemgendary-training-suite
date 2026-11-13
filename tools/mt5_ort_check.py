"""Run an MT5 ONNX artefact under the MetaTrader 5 ONNX Runtime release (manual or test check).

MetaTrader 5 bundles ONNX Runtime 1.17 (build 4230 and later). The suite's venv carries a
newer runtime, and one environment cannot hold two ``onnxruntime`` releases, so this script
runs in a separate interpreter that has ``onnxruntime==1.17.3`` and ``numpy<2`` only (no suite
imports). Set it up once::

    python -m venv .ort117
    .ort117/Scripts/python -m pip install "onnxruntime==1.17.3" "numpy<2"

Manual check of an exported model::

    .ort117/Scripts/python tools/mt5_ort_check.py ForexPredictor.onnx

With ``inputs.npz`` it feeds those arrays (named like the graph inputs) and writes the outputs
to ``outputs.npz``; without it, it feeds zeros at batch 1, like the EA. The test
``tests/unit/test_mt5_onnx_compat.py`` runs it when ``LEMTRAIN_MT5_ORT_PYTHON`` names that
interpreter. Exit status 0 means the runtime loaded and ran the model.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import onnxruntime as ort

_DESCRIPTION = "Run a single-file ONNX model once under the installed ONNX Runtime."
_NUMPY_TYPES = {"tensor(float)": np.float32, "tensor(int64)": np.int64}


def _zero_feed(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    """Zeros for every input at batch 1 (symbolic axes become 1)."""
    feed: dict[str, np.ndarray] = {}
    for arg in session.get_inputs():
        shape = [dim if isinstance(dim, int) else 1 for dim in arg.shape]
        feed[arg.name] = np.zeros(shape, dtype=_NUMPY_TYPES[arg.type])
    return feed


def main(argv: list[str] | None = None) -> int:
    """Load ``model`` on the CPU provider, run it once, print the runtime and output shapes."""
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("model", help="single-file ONNX model")
    parser.add_argument("inputs", nargs="?", help="optional .npz of named inputs")
    parser.add_argument("outputs", nargs="?", help="optional .npz to write the outputs to")
    args = parser.parse_args(argv)

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    if args.inputs:
        with np.load(args.inputs) as data:
            feed = {name: data[name] for name in data.files}
    else:
        feed = _zero_feed(session)
    names = [arg.name for arg in session.get_outputs()]
    results = [np.asarray(value) for value in session.run(names, feed)]
    print(f"onnxruntime {ort.__version__}: loaded {args.model}")
    for name, value in zip(names, results):
        print(f"  {name}: shape {list(value.shape)} dtype {value.dtype}")
    if args.outputs:
        arrays: dict[str, Any] = dict(zip(names, results))
        np.savez(args.outputs, **arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main())
