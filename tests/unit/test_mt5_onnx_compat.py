"""MT5 ONNX artefact and Expert Advisor (R4-C1).

MetaTrader 5 runs ONNX Runtime 1.17 (IR 9 at most) and loads one file per model. The Forex
FP16 release ``<Stem>.onnx`` (FP32 I/O) is stamped IR 9 with embedded weights; opset 18 stays.

The structural tests always run. ``test_mt5_artefact_runs_on_ort_1_17`` runs the artefact
under the real MT5 runtime release when ``LEMTRAIN_MT5_ORT_PYTHON`` names an interpreter that
has ``onnxruntime==1.17.3`` and ``numpy<2`` (one venv cannot hold two ONNX Runtime releases;
setup in ``tools/mt5_ort_check.py``). Without it the same check is the manual step documented
there.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from export import mt5_signal as mt5_signal_cli
from models.forex_predictor import ForexPredictor
from training.export.common import ExportError
from training.export.forex_exporter import ForexModelExporter
from training.export.mql5_ea import ea_spec, render_mql5_ea
from training.export.mt5_compat import (
    MT5_MAX_IR_VERSION,
    MT5_ORT_VERSION,
    ir9_blockers,
    make_mt5_compatible,
    stamp_mt5_ir,
)
from training.export.mt5_signal import export_forex_onnx_from_model, forex_onnx_graph
from training.export.onnx_weights import sidecar_path
from training.registry import get_model_registry

ORT_PYTHON_ENV = "LEMTRAIN_MT5_ORT_PYTHON"
_ROOT = Path(__file__).resolve().parents[2]
_CHECK_SCRIPT = _ROOT / "tools" / "mt5_ort_check.py"
_TIMEFRAMES = [1, 5, 15, 60, 240, 1440]


@pytest.fixture(name="forex_model", scope="module")
def fixture_forex_model() -> ForexPredictor:
    """A random-weight ForexPredictor with the registry's six timeframes."""
    torch.manual_seed(0)
    return ForexPredictor(active_timeframes=list(_TIMEFRAMES)).eval()


@pytest.fixture(name="release", scope="module")
def fixture_release(
    forex_model: ForexPredictor, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Path]:
    """The Forex release files written by the production exporter."""
    out = tmp_path_factory.mktemp("forex_release")
    return ForexModelExporter().export(forex_model, "forex_predictor", output_dir=out)


def _assert_single_file_ir9(path: Path) -> onnx.ModelProto:
    """Loads ``path`` and checks the MT5 constraints; returns the model."""
    assert not sidecar_path(path).exists()
    model = onnx.load(str(path), load_external_data=False)
    assert model.ir_version <= MT5_MAX_IR_VERSION
    assert {(o.domain, o.version) for o in model.opset_import} == {("", 18)}
    assert not model.functions
    assert not ir9_blockers(model)
    assert all(t.data_location != TensorProto.EXTERNAL for t in model.graph.initializer)
    onnx.checker.check_model(model, full_check=True)
    return model


def test_fp16_release_is_the_mt5_artefact(release: dict[str, Path]) -> None:
    """``<Stem>.onnx``: IR 9, opset 18, embedded weights, FP16 weights behind FP32 I/O."""
    assert "onnx_fp16_data" not in release
    model = _assert_single_file_ir9(release["onnx_fp16"])
    inputs = {v.name: v.type.tensor_type.elem_type for v in model.graph.input}
    outputs = {v.name: v.type.tensor_type.elem_type for v in model.graph.output}
    assert inputs.pop("pair_idx") == TensorProto.INT64
    assert set(inputs) == {f"tf_{tf}" for tf in _TIMEFRAMES}
    assert set(inputs.values()) == {TensorProto.FLOAT}
    assert outputs == {"direction_logits": TensorProto.FLOAT, "magnitude": TensorProto.FLOAT}
    assert TensorProto.FLOAT16 in {t.data_type for t in model.graph.initializer}


def test_fp32_release_keeps_its_sidecar(release: dict[str, Path]) -> None:
    """The FP32 release is for other runtimes: unchanged (sidecar, exporter IR)."""
    assert release["onnx_fp32_data"].is_file()


def test_fp32_mt5_export_is_one_ir9_file(forex_model: ForexPredictor, tmp_path: Path) -> None:
    """``export_forex_onnx_from_model`` writes one MT5-loadable FP32 file."""
    path = export_forex_onnx_from_model(forex_model, tmp_path / "forex_mt5.onnx", _TIMEFRAMES)
    model = _assert_single_file_ir9(path)
    assert TensorProto.FLOAT16 not in {t.data_type for t in model.graph.initializer}


def _tiny_model(**extra: object) -> onnx.ModelProto:
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], **extra)


def test_stamp_lowers_the_ir_version_only() -> None:
    """IR 10 becomes 9; opset and graph are kept."""
    model = _tiny_model(ir_version=10)
    stamped = stamp_mt5_ir(model)
    assert stamped.ir_version == 9
    assert stamped.opset_import[0].version == 18


def test_stamp_refuses_ir10_features() -> None:
    """4-bit tensors, local functions and opsets above 20 cannot run on ORT 1.17."""
    int4 = _tiny_model(ir_version=10)
    int4.graph.initializer.append(helper.make_tensor("w", TensorProto.INT4, [2], [1, 2]))
    with pytest.raises(ExportError, match="unsupported type"):
        stamp_mt5_ir(int4)
    func = _tiny_model(ir_version=10)
    func.functions.append(helper.make_function("local", "f", ["a"], ["b"], [], []))
    with pytest.raises(ExportError, match="local function"):
        stamp_mt5_ir(func)
    opset = _tiny_model(ir_version=10)
    opset.opset_import[0].version = 21
    with pytest.raises(ExportError, match="opset 21"):
        stamp_mt5_ir(opset)


def test_make_compatible_refuses_a_sidecar(tmp_path: Path) -> None:
    """MT5 ``OnnxCreate`` cannot read ``.onnx.data``."""
    path = tmp_path / "m.onnx"
    onnx.save_model(_tiny_model(ir_version=10), str(path))
    sidecar_path(path).write_bytes(b"0")
    with pytest.raises(ExportError, match="sidecar"):
        make_mt5_compatible(path)


@pytest.mark.skipif(
    not os.environ.get(ORT_PYTHON_ENV),
    reason=f"{ORT_PYTHON_ENV} names no onnxruntime=={MT5_ORT_VERSION} interpreter",
)
def test_mt5_artefact_runs_on_ort_1_17(
    release: dict[str, Path], forex_model: ForexPredictor, tmp_path: Path
) -> None:
    """The FP16 artefact loads and matches PyTorch under MT5's ONNX Runtime release."""
    fx = forex_onnx_graph(forex_model, _TIMEFRAMES)
    torch.manual_seed(1)
    sample = tuple(
        torch.randn(1, *t.shape[1:]) if t.is_floating_point() else torch.tensor([3])
        for t in fx.inputs
    )
    names = [f"tf_{tf}" for tf in _TIMEFRAMES] + ["pair_idx"]
    outputs = tmp_path / "outputs.npz"
    np.savez(
        tmp_path / "inputs.npz",
        allow_pickle=False,
        **{n: t.numpy() for n, t in zip(names, sample, strict=True)},
    )
    result = subprocess.run(
        [
            os.environ[ORT_PYTHON_ENV],
            "-I",
            str(_CHECK_SCRIPT),
            str(release["onnx_fp16"]),
            str(tmp_path / "inputs.npz"),
            str(outputs),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"onnxruntime {MT5_ORT_VERSION}" in result.stdout
    with torch.no_grad():
        expected = [t.numpy() for t in fx.module(*sample)]
    with np.load(outputs) as got:
        for want, name in zip(expected, ("direction_logits", "magnitude"), strict=True):
            scale = float(np.abs(want).max()) or 1.0
            assert float(np.abs(got[name] - want).max()) <= 2e-2 * scale


def _registry_info() -> dict[str, object]:
    spec = get_model_registry().get_model("forex_predictor")
    assert spec is not None
    return dict(spec.raw_config)


def test_ea_is_mql5_with_onnx_and_ctrade() -> None:
    """The EA uses MQL5 APIs (no MQL4 OrderSend/OP_BUY) and the model's real shapes."""
    source = render_mql5_ea(ea_spec(_registry_info(), "ForexPredictor.onnx"))
    assert "#include <Trade\\Trade.mqh>" in source
    assert "CTrade   g_trade;" in source
    assert "OnnxCreate(ModelFile, ONNX_DEFAULT)" in source
    assert "OnnxRelease(g_model)" in source
    assert "OP_BUY" not in source
    assert "OrderSend" not in source
    assert "ulong shape_0[] = {1, 512, 14};" in source
    assert "ulong shape_5[] = {1, 252, 14};" in source
    assert "OnnxSetInputShape(g_model, TF_COUNT, pair_shape)" in source
    assert "ulong direction_shape[] = {1, 3};" in source
    assert (
        "OnnxRun(g_model, ONNX_DEFAULT, x_0, x_1, x_2, x_3, x_4, x_5, pair, direction, "
        "magnitude)" in source
    )
    assert "matrixf x_0(TF_LOOKBACK[0], FEATURES);" in source
    assert 'input string ModelFile     = "ForexPredictor.onnx";' in source
    assert source.count("{") == source.count("}")


def test_ea_risk_default_comes_from_the_registry() -> None:
    """``RiskPct`` is ``evaluation.risk_per_trade`` in percent."""
    info = _registry_info()
    assert "input double RiskPct       = 1.00;" in render_mql5_ea(ea_spec(info, "m.onnx"))
    info["evaluation"] = {"risk_per_trade": 0.02}
    assert "input double RiskPct       = 2.00;" in render_mql5_ea(ea_spec(info, "m.onnx"))


def test_ea_unit_prices_follow_the_label_scaling() -> None:
    """Broker aliases share the canonical index and model-unit price."""
    spec = ea_spec(_registry_info(), "m.onnx")
    by_name = {s.name: s for s in spec.symbols}
    assert by_name["USTEC"].pair_index == by_name["NAS100"].pair_index
    assert by_name["USTEC"].unit_price == by_name["NAS100"].unit_price
    assert by_name["EURUSD"].unit_price == pytest.approx(0.0001)
    assert by_name["XAUUSD"].unit_price == pytest.approx(1.0)


def test_cli_writes_the_ea(tmp_path: Path) -> None:
    """``python -m export.mt5_signal --mode mql5_ea`` writes the rendered EA."""
    out = tmp_path / "ea" / "LemGendaryForexEA.mq5"
    argv = [sys.argv[0], "--mode", "mql5_ea", "--out", str(out)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        mt5_signal_cli.main()
    assert "OnnxRun(" in out.read_text(encoding="utf-8")
