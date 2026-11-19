"""Schema validator for metrics.csv and history.csv."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from training.telemetry.schemas import (
    COMMON_HISTORY_COLUMNS,
    COMMON_STEP_COLUMNS,
    FACE_DETECTION_METRIC_COLUMNS,
    FOREX_METRIC_COLUMNS,
    LLM_METRIC_COLUMNS,
    VISION_METRIC_COLUMNS,
    YOLO_METRIC_COLUMNS,
    TaskFamily,
)

logger = logging.getLogger("lemtrain.telemetry.validator")


class TelemetrySchemaError(ValueError):
    """Raised when metrics.csv or history.csv does not conform to the schema."""


def _check_row_numbers(row: dict[str, Any], idx: int) -> None:
    """Validate numeric columns for a step-level telemetry row."""
    for num_col in ("step", "epoch"):
        val = row.get(num_col)
        if val is None or not str(val).isdigit():
            raise TelemetrySchemaError(
                f"Row {idx}: column '{num_col}' must be an integer, got {val!r}"
            )

    for float_col in ("loss", "lr", "grad_norm", "throughput", "step_time"):
        f_val = row.get(float_col)
        if f_val is not None and str(f_val).strip() != "":
            try:
                float(str(f_val))
            except ValueError as err:
                raise TelemetrySchemaError(
                    f"Row {idx}: column '{float_col}' must be a float, got {f_val!r}"
                ) from err


def _check_history_numbers(row: dict[str, Any], idx: int) -> None:
    """Validate numeric columns for an epoch-level history row."""
    val = row.get("epoch")
    if val is None or not str(val).isdigit():
        raise TelemetrySchemaError(
            f"Row {idx}: column 'epoch' must be an integer, got {val!r}"
        )

    for float_col in ("train_loss", "val_loss", "lr"):
        f_val = row.get(float_col)
        if f_val is not None and str(f_val).strip() != "":
            try:
                float(str(f_val))
            except ValueError as err:
                raise TelemetrySchemaError(
                    f"Row {idx}: column '{float_col}' must be a float, got {f_val!r}"
                ) from err


class TelemetryCsvValidator:
    """Validates structural and type integrity of step and epoch telemetry files."""

    @staticmethod
    def get_expected_metrics_columns(family: TaskFamily | str) -> list[str]:
        """Return the list of expected columns for step-level metrics.csv."""
        family_str = family.value if isinstance(family, TaskFamily) else str(family)
        base = list(COMMON_STEP_COLUMNS)
        if family_str == TaskFamily.FOREX.value:
            extra = ["phase", "fold", "pairs"]
        elif family_str == TaskFamily.YOLO.value:
            extra = ["box_loss", "cls_loss", "dfl_loss"]
        elif family_str == TaskFamily.LLM.value:
            extra = ["eval_loss", "perplexity"]
        else:
            extra = ["psnr", "ssim"]
        return base + extra

    @staticmethod
    def get_expected_history_columns(family: TaskFamily | str) -> list[str]:
        """Return the list of expected columns for epoch-level history.csv."""
        family_str = family.value if isinstance(family, TaskFamily) else str(family)
        base = list(COMMON_HISTORY_COLUMNS)
        if family_str == TaskFamily.FOREX.value:
            family_cols = list(FOREX_METRIC_COLUMNS)
        elif family_str == TaskFamily.YOLO.value:
            family_cols = list(YOLO_METRIC_COLUMNS)
        elif family_str == TaskFamily.FACE_DETECTION.value:
            family_cols = list(FACE_DETECTION_METRIC_COLUMNS)
        elif family_str == TaskFamily.LLM.value:
            family_cols = list(LLM_METRIC_COLUMNS)
        else:
            family_cols = list(VISION_METRIC_COLUMNS)
        return base + family_cols

    @classmethod
    def validate_metrics_csv(
        cls,
        file_path: Path | str,
        family: TaskFamily | str = TaskFamily.VISION,
        strict_columns: bool = False,
    ) -> list[dict[str, Any]]:
        """Validate step-level metrics.csv.

        Args:
            file_path: Path to the metrics.csv file.
            family: Model task family.
            strict_columns: If True, columns must match expected list exactly.

        Returns:
            List of parsed and validated row dictionaries.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"metrics.csv not found at {path}")

        expected = (
            cls.get_expected_metrics_columns(family)
            if strict_columns
            else list(COMMON_STEP_COLUMNS)
        )
        rows: list[dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise TelemetrySchemaError(f"Missing header in metrics file: {path}")

            actual_fields = set(reader.fieldnames)
            missing = set(expected) - actual_fields
            if missing:
                raise TelemetrySchemaError(
                    f"metrics.csv at {path} is missing mandatory columns: {sorted(missing)}"
                )

            for idx, r in enumerate(reader, start=1):
                _check_row_numbers(r, idx)
                rows.append(dict(r))

        return rows

    @classmethod
    def validate_history_csv(
        cls,
        file_path: Path | str,
        family: TaskFamily | str = TaskFamily.VISION,
        strict_columns: bool = False,
    ) -> list[dict[str, Any]]:
        """Validate epoch-level history.csv.

        Args:
            file_path: Path to the history.csv file.
            family: Model task family.
            strict_columns: If True, columns must match expected list exactly.

        Returns:
            List of parsed and validated row dictionaries.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"history.csv not found at {path}")

        expected = (
            cls.get_expected_history_columns(family)
            if strict_columns
            else ["epoch", "timestamp", "train_loss", "val_loss", "lr"]
        )
        rows: list[dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise TelemetrySchemaError(f"Missing header in history file: {path}")

            actual_fields = set(reader.fieldnames)
            missing = set(expected) - actual_fields
            if missing:
                raise TelemetrySchemaError(
                    f"history.csv at {path} is missing mandatory columns: {sorted(missing)}"
                )

            for idx, r in enumerate(reader, start=1):
                _check_history_numbers(r, idx)
                rows.append(dict(r))

        return rows
