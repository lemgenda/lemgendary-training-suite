"""Atomic and crash-safe CSV recorder with temp-file rename guarantees."""

from __future__ import annotations

import csv
import io
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from training.checkpoint.manager import _resilient_replace
from training.utils.atomic import remove_temp_file_safely

logger = logging.getLogger("lemtrain.telemetry.recorder")


class AtomicCsvRecorder:
    """Thread-safe and crash-safe CSV recorder supporting atomic append and upsert."""

    def __init__(self, file_path: Path | str, fieldnames: Sequence[str]) -> None:
        self.file_path = Path(file_path).resolve()
        self.fieldnames = list(fieldnames)
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Initialize the CSV with header if missing or empty."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            self._write_new_file([])

    def _write_new_file(self, rows: list[dict[str, Any]]) -> None:
        """Atomically create a new file with headers and initial rows."""
        parent_dir = self.file_path.parent
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f"{self.file_path.stem}_", suffix=".tmp", dir=str(parent_dir)
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            _resilient_replace(tmp_path, self.file_path)
        finally:
            remove_temp_file_safely(tmp_path)

    def append_row(self, row: dict[str, Any]) -> None:
        """Atomically append a single row to the CSV file using temp-and-replace."""
        self.append_rows([row])

    def append_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        """Atomically append multiple rows crash-safely."""
        if not rows:
            return

        parent_dir = self.file_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        existing_content = b""
        if self.file_path.exists():
            try:
                existing_content = self.file_path.read_bytes()
            except OSError as exc:
                logger.warning("Could not read %s for append: %s", self.file_path, exc)

        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f"{self.file_path.stem}_", suffix=".tmp", dir=str(parent_dir)
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "wb") as f_bin:
                if existing_content:
                    f_bin.write(existing_content)
                    if not existing_content.endswith(b"\n"):
                        f_bin.write(b"\n")
                else:
                    # Write header if starting empty
                    header_line = ",".join(self.fieldnames) + "\n"
                    f_bin.write(header_line.encode("utf-8"))

                # Write rows using text wrapper
                out_str = io.StringIO()
                writer = csv.DictWriter(out_str, fieldnames=self.fieldnames, extrasaction="ignore")
                for r in rows:
                    writer.writerow(r)
                f_bin.write(out_str.getvalue().encode("utf-8"))

                f_bin.flush()
                os.fsync(f_bin.fileno())

            _resilient_replace(tmp_path, self.file_path)
        finally:
            remove_temp_file_safely(tmp_path)

    def upsert_by_key(self, row: dict[str, Any], key_fields: tuple[str, ...]) -> None:
        """Insert or replace an existing row keyed by key_fields atomically."""
        parent_dir = self.file_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        existing_rows: list[dict[str, str]] = []
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f_in:
                    existing_rows = list(csv.DictReader(f_in))
            except (OSError, csv.Error) as exc:
                logger.warning("Failed reading existing rows from %s: %s", self.file_path, exc)

        row_key = tuple(str(row.get(k, "")).strip() for k in key_fields)
        matched = False
        updated_rows: list[dict[str, Any]] = []

        for r in existing_rows:
            curr_key = tuple(str(r.get(k, "")).strip() for k in key_fields)
            if curr_key == row_key:
                updated_rows.append(row)
                matched = True
            else:
                updated_rows.append(r)

        if not matched:
            updated_rows.append(row)

        self._write_new_file(updated_rows)

    def read_all(self) -> list[dict[str, str]]:
        """Read all rows from the CSV file."""
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            return []
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
