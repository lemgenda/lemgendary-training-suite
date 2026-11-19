"""Unit tests for telemetry canonical schemas, recorders, and validators."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from training.telemetry import (
    COMMON_HISTORY_COLUMNS,
    COMMON_STEP_COLUMNS,
    AtomicCsvRecorder,
    HistoryRecord,
    StepRecord,
    TaskFamily,
    TelemetryCsvValidator,
    TelemetrySchemaError,
)


def test_step_record_to_dict_conforms_to_common_step_columns() -> None:
    """Verify StepRecord outputs all required COMMON_STEP_COLUMNS."""
    rec = StepRecord(
        step=10,
        epoch=1,
        timestamp=1700000000.1234,
        loss=0.456789,
        lr=0.0001,
        grad_norm=1.2345,
        throughput=34.56,
        step_time=0.029,
        vram_gb=4.512,
        is_spike=False,
        is_non_finite=False,
        governor_action="none",
        extra_metrics={"psnr": 28.5},
    )
    d = rec.to_dict()
    for col in COMMON_STEP_COLUMNS:
        assert col in d
    assert d["step"] == 10
    assert d["epoch"] == 1
    assert d["psnr"] == "28.500000"


def test_history_record_to_dict_conforms_to_common_history_columns() -> None:
    """Verify HistoryRecord outputs all required COMMON_HISTORY_COLUMNS."""
    rec = HistoryRecord(
        epoch=5,
        timestamp=1700000100.0,
        train_loss=0.35,
        val_loss=0.32,
        lr=0.00005,
        grad_norm=0.98,
        throughput=40.2,
        memory_allocated_gb=4.2,
        memory_peak_gb=6.1,
        wall_time_sec=120.5,
        governor_action="propel",
        is_spike=False,
        is_non_finite=False,
        is_best=True,
        quality_score=35.2,
        checkpoint_path="checkpoints/best.pth",
        family=TaskFamily.VISION,
        family_metrics={"psnr": 31.4, "ssim": 0.92},
    )
    d = rec.to_dict()
    for col in COMMON_HISTORY_COLUMNS:
        assert col in d
    assert d["psnr"] == "31.4000"
    assert d["ssim"] == "0.9200"


def test_atomic_csv_recorder_creates_file_and_appends(tmp_path: Path) -> None:
    """Verify AtomicCsvRecorder creates headers and appends records atomically."""
    target_csv = tmp_path / "sub" / "metrics.csv"
    recorder = AtomicCsvRecorder(target_csv, fieldnames=COMMON_STEP_COLUMNS)

    assert target_csv.exists()
    rows = recorder.read_all()
    assert len(rows) == 0

    step_row = StepRecord(
        step=1,
        epoch=1,
        timestamp=1700000000.0,
        loss=0.5,
        lr=0.001,
        grad_norm=1.0,
        throughput=20.0,
        step_time=0.05,
        vram_gb=4.0,
    ).to_dict()
    recorder.append_row(step_row)

    rows = recorder.read_all()
    assert len(rows) == 1
    assert rows[0]["step"] == "1"
    assert rows[0]["loss"] == "0.500000"


def test_atomic_csv_recorder_upsert_by_key(tmp_path: Path) -> None:
    """Verify AtomicCsvRecorder updates matching rows in place without duplicates."""
    target_csv = tmp_path / "history.csv"
    recorder = AtomicCsvRecorder(target_csv, fieldnames=["epoch", "val_loss", "lr"])

    recorder.append_row({"epoch": "1", "val_loss": "0.5", "lr": "0.01"})
    recorder.append_row({"epoch": "2", "val_loss": "0.4", "lr": "0.01"})

    # Upsert epoch 2
    recorder.upsert_by_key({"epoch": "2", "val_loss": "0.35", "lr": "0.005"}, key_fields=("epoch",))

    rows = recorder.read_all()
    assert len(rows) == 2
    assert rows[1]["epoch"] == "2"
    assert rows[1]["val_loss"] == "0.35"
    assert rows[1]["lr"] == "0.005"


def test_validator_detects_valid_and_invalid_metrics_csv(tmp_path: Path) -> None:
    """Verify TelemetryCsvValidator accepts valid files and catches missing/bad columns."""
    valid_csv = tmp_path / "valid_metrics.csv"
    rec = AtomicCsvRecorder(valid_csv, fieldnames=COMMON_STEP_COLUMNS)
    rec.append_row(
        StepRecord(
            step=1,
            epoch=1,
            timestamp=1700000000.0,
            loss=0.5,
            lr=0.001,
            grad_norm=1.0,
            throughput=20.0,
            step_time=0.05,
            vram_gb=4.0,
        ).to_dict()
    )

    validated = TelemetryCsvValidator.validate_metrics_csv(valid_csv)
    assert len(validated) == 1

    # Missing mandatory column
    invalid_csv = tmp_path / "invalid_metrics.csv"
    with open(invalid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss"])
        writer.writerow(["1", "1", "0.5"])

    with pytest.raises(TelemetrySchemaError) as exc_info:
        TelemetryCsvValidator.validate_metrics_csv(invalid_csv)
    assert "missing mandatory columns" in str(exc_info.value)

    # Bad non-numeric step
    bad_step_csv = tmp_path / "bad_step.csv"
    bad_cols = list(COMMON_STEP_COLUMNS)
    with open(bad_step_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bad_cols)
        writer.writeheader()
        row = StepRecord(
            step=1,
            epoch=1,
            timestamp=1700000000.0,
            loss=0.5,
            lr=0.001,
            grad_norm=1.0,
            throughput=20.0,
            step_time=0.05,
            vram_gb=4.0,
        ).to_dict()
        row["step"] = "invalid_int"
        writer.writerow(row)

    with pytest.raises(TelemetrySchemaError) as exc_info:
        TelemetryCsvValidator.validate_metrics_csv(bad_step_csv)
    assert "must be an integer" in str(exc_info.value)


def test_validator_detects_valid_and_invalid_history_csv(tmp_path: Path) -> None:
    """Verify TelemetryCsvValidator verifies history.csv epoch files correctly."""
    valid_csv = tmp_path / "valid_history.csv"
    rec = AtomicCsvRecorder(valid_csv, fieldnames=COMMON_HISTORY_COLUMNS)
    rec.append_row(
        HistoryRecord(
            epoch=1,
            timestamp=1700000000.0,
            train_loss=0.5,
            val_loss=0.4,
            lr=0.001,
            grad_norm=1.0,
            throughput=25.0,
            memory_allocated_gb=4.0,
            memory_peak_gb=5.0,
            wall_time_sec=60.0,
        ).to_dict()
    )

    validated = TelemetryCsvValidator.validate_history_csv(valid_csv)
    assert len(validated) == 1

    # Missing mandatory epoch/loss
    invalid_csv = tmp_path / "invalid_history.csv"
    with open(invalid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "timestamp"])
        writer.writerow(["1", "1700000000.0"])

    with pytest.raises(TelemetrySchemaError) as exc_info:
        TelemetryCsvValidator.validate_history_csv(invalid_csv)
    assert "missing mandatory columns" in str(exc_info.value)
