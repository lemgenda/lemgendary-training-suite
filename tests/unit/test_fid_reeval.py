"""FID re-evaluation tool (L12): invariants, ledger and CLI on CPU with a fake FID backend.

``FakeFid`` replaces ``torchmetrics.image.fid.FrechetInceptionDistance`` inside
``training.training.val_metrics``, so no Inception weights are downloaded. It keeps the parts the
tool relies on: the ``feature`` argument, ``real_features_sum`` and ``update``/``compute``.
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from training.checkpoint.lifecycle_manager import CheckpointLifecycleManager
from training.server.routes.gui_model_stats import model_with_stats

from training.fid_reeval import cli, inputs, ledger
from training.fid_reeval.measure import (
    FID_FEATURE_DIM,
    FID_MAX_CHUNK,
    ChunkGuard,
    FidInvariantError,
    FidUnavailableError,
    measure_fid,
)
from training.training import val_metrics

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "training" / "fid_reeval"
CPU = torch.device("cpu")


class FakeFid:
    """Tiny deterministic stand-in for the torchmetrics FID module."""

    instances: ClassVar[list[FakeFid]] = []

    def __init__(self, feature: int = 2048, **_kwargs) -> None:
        gen = torch.Generator().manual_seed(0)
        self.proj = torch.randn(48, feature, generator=gen)
        self.real_features_sum = torch.zeros(feature, dtype=torch.float64)
        self.real: list[torch.Tensor] = []
        self.fake: list[torch.Tensor] = []
        self.chunks: list[int] = []
        FakeFid.instances.append(self)

    def to(self, _device):
        """Device move (no-op)."""
        return self

    def eval(self):
        """Eval mode (no-op)."""
        return self

    def update(self, imgs: torch.Tensor, real: bool) -> None:
        """Project pooled uint8 images to ``feature`` dimensions."""
        assert imgs.dtype == torch.uint8
        self.chunks.append(imgs.size(0))
        feats = F.adaptive_avg_pool2d(imgs.float() / 255.0, (4, 4)).flatten(1) @ self.proj
        (self.real if real else self.fake).append(feats)
        if real:
            self.real_features_sum += feats.sum(0).double()

    def compute(self) -> torch.Tensor:
        """Squared distance of the feature means (a fake, but zero for identical sets)."""
        return ((torch.cat(self.real).mean(0) - torch.cat(self.fake).mean(0)) ** 2).sum()


class FakeFid64(FakeFid):
    """A backend that ignores ``feature`` and uses the old 64-dim layer."""

    def __init__(self, feature: int = 2048, **kwargs) -> None:
        del feature
        super().__init__(feature=64, **kwargs)


class Identity(torch.nn.Module):
    """Perfect restorer with one weight so checkpoints have a state dict."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input scaled by ``scale``."""
        return x * self.scale


class Noisy(Identity):
    """Restorer that adds a fixed offset (FID must be positive)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a brightened input."""
        return torch.clamp(x * self.scale + 0.3, 0.0, 1.0)


class Shrinking(Identity):
    """Restorer whose output size differs from the target (never reaches FID)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a half-size output."""
        return x[..., ::2, ::2]


class TinyValSet(torch.utils.data.Dataset):
    """``n`` random (input, target) pairs; input == target."""

    def __init__(self, n: int) -> None:
        gen = torch.Generator().manual_seed(1)
        self.items = [torch.rand(3, 16, 16, generator=gen) for _ in range(n)]
        self.path_cache = {"LemGendizedTiny": "/data/LemGendizedTiny"}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index], self.items[index]


@pytest.fixture(name="fake_fid")
def fixture_fake_fid(monkeypatch: pytest.MonkeyPatch) -> type[FakeFid]:
    """Install :class:`FakeFid` as the validation FID backend."""
    FakeFid.instances.clear()
    monkeypatch.setattr(val_metrics, "FrechetInceptionDistance", FakeFid)
    return FakeFid


def _batches(n: int, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    loader = torch.utils.data.DataLoader(TinyValSet(n), batch_size=batch_size)
    return list(loader)


# --- measurement and invariants -------------------------------------------------------------


def test_identity_model_gives_zero_fid_with_small_chunks(fake_fid: type[FakeFid]) -> None:
    """Five-image batches reach the backend in chunks of at most 2, at 2048 features."""
    result = measure_fid(Identity(), _batches(7, 5), CPU)
    assert result.fid == pytest.approx(0.0, abs=1e-9)
    assert (result.n_images, result.batches) == (7, 2)
    assert result.feature_dim == FID_FEATURE_DIM
    assert result.max_chunk == FID_MAX_CHUNK
    assert max(fake_fid.instances[0].chunks) <= FID_MAX_CHUNK


def test_degraded_model_gives_positive_fid(fake_fid: type[FakeFid]) -> None:
    """A model that changes the images has a non-zero distance."""
    del fake_fid
    assert measure_fid(Noisy(), _batches(4, 2), CPU).fid > 0.0


def test_feature_size_other_than_2048_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 64-dim layer behind the old fabricated values cannot be used."""
    monkeypatch.setattr(val_metrics, "FrechetInceptionDistance", FakeFid64)
    with pytest.raises(FidInvariantError, match="64"):
        measure_fid(Identity(), _batches(4, 2), CPU)


def test_chunk_guard_refuses_three_images() -> None:
    """A chunk above the ROBOTS.md limit raises and is not forwarded."""
    backend = FakeFid()
    guard = ChunkGuard(backend)
    with pytest.raises(FidInvariantError, match="limit of 2"):
        guard.update(torch.zeros(3, 3, 8, 8, dtype=torch.uint8), real=True)
    assert not backend.chunks


def test_invariant_error_is_not_swallowed_by_the_validation_updater() -> None:
    """``_update_fid`` logs and skips METRIC_ERRORS; the invariant error must not be one."""
    assert not issubclass(FidInvariantError, val_metrics.METRIC_ERRORS)


def test_batches_that_miss_fid_abort_the_run(fake_fid: type[FakeFid]) -> None:
    """A shape mismatch would silently drop batches from FID; the tool refuses instead."""
    del fake_fid
    with pytest.raises(FidInvariantError, match="partial FID"):
        measure_fid(Shrinking(), _batches(4, 2), CPU)


def test_single_image_is_refused(fake_fid: type[FakeFid]) -> None:
    """FID needs a covariance estimate."""
    del fake_fid
    with pytest.raises(FidInvariantError, match="at least 2"):
        measure_fid(Identity(), _batches(1, 1), CPU)


def test_missing_backend_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without torchmetrics the tool fails loudly instead of returning a value."""
    monkeypatch.setattr(val_metrics, "FrechetInceptionDistance", None)
    with pytest.raises(FidUnavailableError, match="torchmetrics"):
        measure_fid(Identity(), _batches(2, 2), CPU)


# --- ledger and mark-unverified ---------------------------------------------------------------

METRICS_HEADER = "Epoch,Train_Loss,Val_Loss,PSNR,LPIPS,FID,Quality_Score\n"


def _write_metrics(hub: Path) -> Path:
    hub.mkdir(parents=True, exist_ok=True)
    path = hub / "metrics.csv"
    path.write_text(
        METRICS_HEADER + "1,0.1,0.2,30.0,0.1,2.8500,500.0\n2,0.1,0.2,31.0,0.1,2.7800,501.0\n",
        encoding="utf-8",
    )
    return path


def _ledger(hub: Path) -> list[dict[str, str]]:
    with open(hub / ledger.LEDGER_NAME, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_mark_unverified_appends_once_and_leaves_metrics_untouched(tmp_path: Path) -> None:
    """Two runs give one row per epoch; metrics.csv stays byte-identical."""
    metrics = _write_metrics(tmp_path / "codeformer")
    before = metrics.read_bytes()
    args = ["mark-unverified", "--model", "codeformer", "--models-root", str(tmp_path)]
    assert cli.main(args) == 0
    assert cli.main(args) == 0
    rows = _ledger(tmp_path / "codeformer")
    assert [(r["epoch"], r["value"], r["status"]) for r in rows] == [
        ("1", "2.8500", "unverified"),
        ("2", "2.7800", "unverified"),
    ]
    assert all(r["source"] == ledger.SOURCE_RECALIBRATED and r["recorded_at"] for r in rows)
    assert metrics.read_bytes() == before


def test_mark_unverified_dry_run_writes_nothing(tmp_path: Path, capsys) -> None:
    """``--dry-run`` reports the count only."""
    _write_metrics(tmp_path / "mirnet_lowlight")
    args = ["mark-unverified", "--all", "--models-root", str(tmp_path), "--dry-run"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "mirnet_lowlight: would mark 2" in out
    assert out.count("skipped") == 3  # the other three models have no metrics.csv
    assert not (tmp_path / "mirnet_lowlight" / ledger.LEDGER_NAME).exists()


def test_mark_unverified_rejects_other_models(tmp_path: Path) -> None:
    """Only the four rewritten models can be marked."""
    _write_metrics(tmp_path / "nafnet_denoising")
    with pytest.raises(SystemExit):
        cli.main(["mark-unverified", "--model", "nafnet_denoising", "--models-root", str(tmp_path)])
    with pytest.raises(ledger.LedgerError):
        ledger.plan_unverified(tmp_path / "nafnet_denoising", "nafnet_denoising")


def test_foreign_ledger_header_is_never_rewritten(tmp_path: Path) -> None:
    """An unexpected header stops the append; the file keeps its content."""
    hub = tmp_path / "ffanet_indoor"
    _write_metrics(hub)
    foreign = hub / ledger.LEDGER_NAME
    foreign.write_text("something,else\n1,2\n", encoding="utf-8")
    args = ["mark-unverified", "--model", "ffanet_indoor", "--models-root", str(tmp_path)]
    assert cli.main(args) == cli.EXIT_INPUT
    assert foreign.read_text(encoding="utf-8") == "something,else\n1,2\n"


# --- measure command ----------------------------------------------------------------------------


@pytest.fixture(name="tiny_stack")
def fixture_tiny_stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Patch model and dataset builders; save an Identity checkpoint for ``ffanet_indoor``."""
    monkeypatch.setattr(inputs, "build_model", lambda model_id, config: Identity())
    monkeypatch.setattr(inputs, "build_val_dataset", lambda *a: TinyValSet(5))
    ckpt_dir = tmp_path / "ffanet_indoor" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save({"epoch": 12, "model_state_dict": Identity().state_dict()}, ckpt_dir / "best.pth")
    return tmp_path


def _measure_args(root: Path, *extra: str) -> list[str]:
    return [
        "measure",
        "--model",
        "ffanet_indoor",
        "--models-root",
        str(root),
        "--device",
        "cpu",
    ] + [*extra]


def test_measure_appends_a_marked_row(fake_fid: type[FakeFid], tiny_stack: Path) -> None:
    """Every run appends one ``reevaluated`` row with full provenance."""
    del fake_fid
    assert cli.main(_measure_args(tiny_stack, "--batch-size", "3", "--note", "owner run")) == 0
    assert cli.main(_measure_args(tiny_stack)) == 0
    rows = _ledger(tiny_stack / "ffanet_indoor")
    assert len(rows) == 2
    row = rows[0]
    assert (row["status"], row["source"], row["metric"]) == ("reevaluated", "reevaluated", "FID")
    assert (row["epoch"], row["n_images"], row["feature_dim"], row["max_chunk"]) == (
        "12",
        "5",
        "2048",
        "2",
    )
    assert float(row["value"]) == pytest.approx(0.0, abs=1e-4)
    assert len(row["checkpoint_sha256"]) == 64 and row["recorded_at"]
    assert "strict load" in row["note"] and "owner run" in row["note"]
    assert "LemGendizedTiny" in row["note"] and "n_images<2048" in row["note"]


def test_measure_dry_run_writes_nothing(fake_fid: type[FakeFid], tiny_stack: Path) -> None:
    """``--dry-run`` validates (checkpoint load, data) without measuring or writing."""
    assert cli.main(_measure_args(tiny_stack, "--dry-run")) == 0
    assert not fake_fid.instances
    assert not (tiny_stack / "ffanet_indoor" / ledger.LEDGER_NAME).exists()


@pytest.mark.parametrize(
    ("argv_tail", "message"),
    [
        (["--checkpoint", "missing.pth"], "checkpoint not found"),
        (["--batch-size", "0"], "--batch-size"),
        (["--datasets-root", "no/such/dir"], "datasets root"),
    ],
)
def test_measure_rejects_bad_arguments(
    fake_fid: type[FakeFid], tiny_stack: Path, capsys, argv_tail: list[str], message: str
) -> None:
    """Bad inputs exit with code 2 and a message, before anything is measured."""
    assert cli.main(_measure_args(tiny_stack, "--dry-run", *argv_tail)) == cli.EXIT_INPUT
    assert message in capsys.readouterr().err
    assert not fake_fid.instances


@pytest.mark.parametrize(
    ("model_id", "message"),
    [("no_such_model", "unknown model"), ("nima_technical", "no 'fid'")],
)
def test_measure_rejects_models_without_fid(
    tiny_stack: Path, capsys, model_id: str, message: str
) -> None:
    """Only registered image models with a ``fid`` target can be measured."""
    argv = ["measure", "--model", model_id, "--models-root", str(tiny_stack), "--dry-run"]
    assert cli.main(argv) == cli.EXIT_INPUT
    assert message in capsys.readouterr().err


def test_strict_load_refuses_a_mismatched_checkpoint(
    fake_fid: type[FakeFid], tiny_stack: Path, capsys
) -> None:
    """Random weights must never be evaluated silently; ``--non-strict`` is explicit."""
    del fake_fid
    ckpt = tiny_stack / "ffanet_indoor" / "checkpoints" / "best.pth"
    torch.save({"epoch": 3, "model_state_dict": {"other.weight": torch.ones(1)}}, ckpt)
    assert cli.main(_measure_args(tiny_stack, "--dry-run")) == cli.EXIT_INPUT
    assert "does not match" in capsys.readouterr().err
    assert cli.main(_measure_args(tiny_stack, "--dry-run", "--non-strict")) == 0


def test_checkpoint_state_strips_ddp_and_compile_prefixes() -> None:
    """``module.`` and ``_orig_mod.`` prefixes are removed."""
    weight = torch.ones(1)
    state = inputs.checkpoint_state({"model_state": {"module._orig_mod.scale": weight}})
    assert list(state) == ["scale"]


def test_status_reports_counts(fake_fid: type[FakeFid], tiny_stack: Path, capsys) -> None:
    """``status`` summarises unverified epochs and the latest re-evaluation."""
    del fake_fid
    _write_metrics(tiny_stack / "ffanet_indoor")
    root = str(tiny_stack)
    assert cli.main(["mark-unverified", "--model", "ffanet_indoor", "--models-root", root]) == 0
    assert cli.main(_measure_args(tiny_stack)) == 0
    assert cli.main(["status", "--model", "ffanet_indoor", "--models-root", root]) == 0
    out = capsys.readouterr().out
    assert "2 unverified epoch(s), 1 re-evaluation(s), latest FID" in out


# --- scope: no metrics.csv writes, no rung state ------------------------------------------------


def test_filter_metrics_row_for_fid_provenance(tmp_path: Path) -> None:
    """Unverified epochs drop FID and Quality_Score; reevaluated epochs restore measured FID."""
    hub = tmp_path / "codeformer"
    _write_metrics(hub)
    assert (
        cli.main(["mark-unverified", "--model", "codeformer", "--models-root", str(tmp_path)]) == 0
    )
    assert ledger.unverified_fid_epochs(hub) == {"1", "2"}

    raw_row = {"Epoch": "1", "PSNR": "30.0", "FID": "2.8500", "Quality_Score": "500.0"}
    filtered = ledger.filter_metrics_row_for_fid_provenance(raw_row, hub)
    assert "FID" not in filtered
    assert "Quality_Score" not in filtered
    assert filtered["PSNR"] == "30.0"

    ledger.append_rows(
        ledger.ledger_path(hub),
        [
            ledger.new_row(
                recorded_at=ledger.utc_now(),
                model_id="codeformer",
                status=ledger.STATUS_REEVALUATED,
                source=ledger.SOURCE_REEVALUATED,
                epoch="1",
                metric="FID",
                value="14.2500",
            )
        ],
    )
    assert ledger.unverified_fid_epochs(hub) == {"2"}
    assert ledger.reevaluated_fid_by_epoch(hub) == {"1": pytest.approx(14.25)}
    after_reeval = ledger.filter_metrics_row_for_fid_provenance(raw_row, hub)
    assert after_reeval["FID"] == "14.2500"
    assert "Quality_Score" not in after_reeval


def test_tool_never_touches_rung_state_or_writes_metrics_csv() -> None:
    """The package imports no governor/ratchet code and calls no metrics.csv writer."""
    for path in PACKAGE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("training.governance", "training.ratchet")
                )
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"_commit_rung", "write_epoch_row", "export_csv"}


def test_filter_metrics_in_lifecycle_manager(tmp_path: Path) -> None:
    """Verify CheckpointLifecycleManager ignores unverified FID and Quality_Score."""

    hub_dir = tmp_path / "LemGendaryModels" / "codeformer"
    hub_dir.mkdir(parents=True)
    metrics_file = hub_dir / "metrics.csv"
    with open(metrics_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Epoch", "PSNR", "FID", "Quality_Score"])
        writer.writeheader()
        writer.writerow({"Epoch": "1", "PSNR": "30.0", "FID": "2.85", "Quality_Score": "500.0"})

    ledger.append_rows(
        ledger.ledger_path(hub_dir),
        [
            ledger.new_row(
                recorded_at=ledger.utc_now(),
                model_id="codeformer",
                status=ledger.STATUS_UNVERIFIED,
                source=ledger.SOURCE_RECALIBRATED,
                epoch="1",
                metric="FID",
                value="2.85",
            )
        ],
    )

    with patch("training.checkpoint.lifecycle_manager.get_model_hub_dir", return_value=hub_dir):
        mgr = CheckpointLifecycleManager(model_name="codeformer", project_root=tmp_path)
        # LifecycleManager initial scan should NOT adopt Quality_Score 500.0 from unverified FID row
        # Since the unverified FID/Quality_Score row is ignored, best_quality_score remains -inf
        assert mgr.tracking.best_quality_score < 0.0


def test_filter_metrics_in_gui_model_stats(tmp_path: Path) -> None:
    """Verify model_with_stats omits unverified FID from history."""

    hub_root = tmp_path / "models_hub"
    hub_dir = hub_root / "codeformer"
    hub_dir.mkdir(parents=True)
    metrics_file = hub_dir / "metrics.csv"
    with open(metrics_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Epoch", "Resolution", "Fraction", "PSNR", "FID"])
        writer.writeheader()
        writer.writerow({
            "Epoch": "1",
            "Resolution": "512",
            "Fraction": "1.0",
            "PSNR": "30.0",
            "FID": "2.85",
        })

    ledger.append_rows(
        ledger.ledger_path(hub_dir),
        [
            ledger.new_row(
                recorded_at=ledger.utc_now(),
                model_id="codeformer",
                status=ledger.STATUS_UNVERIFIED,
                source=ledger.SOURCE_RECALIBRATED,
                epoch="1",
                metric="FID",
                value="2.85",
            )
        ],
    )

    info = {"name": "CodeFormer", "category": "face", "dataset_type": "face"}
    model_stat = model_with_stats("codeformer", info, tmp_path, hub_root, [])
    keys = [d.key for d in model_stat.best_metrics_details]
    assert "fid" not in keys
    assert "psnr" in keys
