"""Append-only FID provenance ledger (torch-free).

One ledger per model, ``LemGendaryModels/<model>/fid_provenance.csv``. Rows are only ever
appended; an existing file whose header differs from :data:`LEDGER_FIELDS` is refused rather than
rewritten. ``metrics.csv`` is read, never written.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("lemtrain.fid_reeval.ledger")

LEDGER_NAME = "fid_provenance.csv"
METRICS_NAME = "metrics.csv"

STATUS_UNVERIFIED = "unverified"
STATUS_REEVALUATED = "reevaluated"
SOURCE_RECALIBRATED = "recalibrate_fid"
SOURCE_REEVALUATED = "reevaluated"

# Models whose metrics.csv FID and Quality_Score columns tools/recalibrate_fid.py rewrote with
# formulas (README v16.9.67, review finding D-03).
FABRICATED_FID_MODELS = ("codeformer", "ffanet_indoor", "mprnet_deraining", "mirnet_lowlight")

UNVERIFIED_NOTE = (
    "FID and Quality_Score rewritten by tools/recalibrate_fid.py (formula, not a measurement); "
    "metrics.csv left unchanged"
)


class LedgerError(ValueError):
    """The ledger or metrics file cannot be used safely."""


# Ledger columns in file order. Empty values mean "not applicable".
LEDGER_FIELDS: tuple[str, ...] = (
    "recorded_at",
    "model_id",
    "status",
    "source",
    "epoch",
    "metric",
    "value",
    "n_images",
    "feature_dim",
    "max_chunk",
    "checkpoint",
    "checkpoint_sha256",
    "datasets_root",
    "device",
    "note",
)
_REQUIRED = LEDGER_FIELDS[:7]

LedgerRow = dict[str, str]
"""One provenance record keyed by :data:`LEDGER_FIELDS`."""


def new_row(**values: str) -> LedgerRow:
    """Ledger row from keyword values; the first seven columns are required, others empty."""
    unknown = set(values) - set(LEDGER_FIELDS)
    missing = [k for k in _REQUIRED if not values.get(k)]
    if unknown or missing:
        raise LedgerError(f"bad ledger row: unknown {sorted(unknown)}, missing {missing}")
    return {k: str(values.get(k, "")) for k in LEDGER_FIELDS}


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string with seconds precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ledger_path(hub_dir: Path) -> Path:
    """Ledger file of the model whose hub directory is ``hub_dir``."""
    return hub_dir / LEDGER_NAME


def read_ledger(path: Path) -> list[LedgerRow]:
    """All rows of the ledger at ``path`` (empty when the file does not exist)."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _check_header(path, reader.fieldnames)
        return [{k: row.get(k) or "" for k in LEDGER_FIELDS} for row in reader]


def _check_header(path: Path, header: Sequence[str] | None) -> None:
    if tuple(header or ()) != LEDGER_FIELDS:
        raise LedgerError(
            f"{path} has an unexpected header {header!r}; refusing to append "
            f"(expected {list(LEDGER_FIELDS)!r}). Move the file aside and inspect it."
        )


def append_rows(path: Path, rows: list[LedgerRow]) -> None:
    """Append ``rows`` to the ledger, creating it with a header when missing."""
    if not rows:
        return
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with open(path, encoding="utf-8", newline="") as f:
            _check_header(path, next(csv.reader(f), None))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def fabricated_fid_rows(metrics_csv: Path) -> list[tuple[str, str]]:
    """``(epoch, fid)`` of every metrics.csv row that carries a FID value."""
    if not metrics_csv.exists():
        raise LedgerError(f"{metrics_csv} does not exist")
    with open(metrics_csv, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if "FID" not in header or "Epoch" not in header:
            raise LedgerError(f"{metrics_csv} has no Epoch/FID columns (header {header!r})")
        return [
            (row["Epoch"].strip(), row["FID"].strip())
            for row in reader
            if (row.get("FID") or "").strip() and (row.get("Epoch") or "").strip()
        ]


def plan_unverified(hub_dir: Path, model_id: str) -> list[LedgerRow]:
    """Ledger rows still needed to mark ``model_id``'s rewritten FID rows as unverified.

    Epochs that already have an ``unverified`` row are skipped, so repeated runs add nothing.
    """
    if model_id not in FABRICATED_FID_MODELS:
        raise LedgerError(
            f"{model_id!r} is not one of the models rewritten by tools/recalibrate_fid.py "
            f"({', '.join(FABRICATED_FID_MODELS)})"
        )
    done = {
        r["epoch"] for r in read_ledger(ledger_path(hub_dir)) if r["status"] == STATUS_UNVERIFIED
    }
    now = utc_now()
    return [
        new_row(
            recorded_at=now,
            model_id=model_id,
            status=STATUS_UNVERIFIED,
            source=SOURCE_RECALIBRATED,
            epoch=epoch,
            metric="FID",
            value=fid,
            note=UNVERIFIED_NOTE,
        )
        for epoch, fid in fabricated_fid_rows(hub_dir / METRICS_NAME)
        if epoch not in done
    ]


@dataclass(frozen=True)
class LedgerStatus:
    """Summary of one model's ledger."""

    model_id: str
    unverified_epochs: int
    reevaluations: int
    latest: LedgerRow | None


def ledger_status(hub_dir: Path, model_id: str) -> LedgerStatus:
    """Count unverified epochs and re-evaluations; ``latest`` is the newest re-evaluation."""
    rows = read_ledger(ledger_path(hub_dir))
    reevaluated = [r for r in rows if r["status"] == STATUS_REEVALUATED]
    return LedgerStatus(
        model_id=model_id,
        unverified_epochs=len({r["epoch"] for r in rows if r["status"] == STATUS_UNVERIFIED}),
        reevaluations=len(reevaluated),
        latest=reevaluated[-1] if reevaluated else None,
    )


def _safe_read_ledger(hub_dir: Path) -> list[LedgerRow]:
    """Return rows of ``hub_dir/fid_provenance.csv`` or ``[]`` on missing/corrupt ledger."""
    try:
        return read_ledger(ledger_path(hub_dir))
    except (OSError, csv.Error, LedgerError):
        return []


def unverified_fid_epochs(hub_dir: Path) -> set[str]:
    """Epochs marked ``unverified`` without a ``reevaluated`` row in ``fid_provenance.csv``."""
    rows = _safe_read_ledger(hub_dir)
    unv = {
        r["epoch"].strip() for r in rows if r.get("status") == STATUS_UNVERIFIED and r.get("epoch")
    }
    reeval = {
        r["epoch"].strip() for r in rows if r.get("status") == STATUS_REEVALUATED and r.get("epoch")
    }
    return unv - reeval


def reevaluated_fid_by_epoch(hub_dir: Path) -> dict[str, float]:
    """Map ``epoch`` -> latest ``reevaluated`` FID value in ``hub_dir/fid_provenance.csv``."""
    out: dict[str, float] = {}
    for row in _safe_read_ledger(hub_dir):
        if row.get("status") != STATUS_REEVALUATED:
            continue
        ep = (row.get("epoch") or "").strip()
        val_str = (row.get("value") or "").strip()
        if not ep or not val_str:
            continue
        try:
            out[ep] = float(val_str)
        except ValueError:
            logger.warning(
                "FID ledger in %s: epoch %s value %r is not a number; row ignored.",
                hub_dir,
                ep,
                val_str,
            )
    return out


def filter_metrics_row_for_fid_provenance(row: Mapping[str, Any], hub_dir: Path) -> dict[str, Any]:
    """Return a copy of a ``metrics.csv`` row with unverified FID/Quality_Score stripped or updated.

    - If the row's epoch is in :func:`unverified_fid_epochs`, ``FID``/``fid`` and
      ``Quality_Score``/``quality_score``/``Quality`` are removed because both columns were
      fabricated by ``tools/recalibrate_fid.py``.
    - If the row's epoch has a ``reevaluated`` entry in :func:`reevaluated_fid_by_epoch`, ``FID``
      is replaced with the re-evaluated value while ``Quality_Score`` remains stripped if that
      epoch was previously marked ``unverified``.
    """
    filtered = dict(row)
    ep = str(filtered.get("Epoch") or filtered.get("epoch") or "").strip()
    if not ep:
        return filtered
    rows = _safe_read_ledger(hub_dir)
    if not rows:
        return filtered
    was_unverified = any(
        (r.get("epoch") or "").strip() == ep and r.get("status") == STATUS_UNVERIFIED for r in rows
    )
    reeval = reevaluated_fid_by_epoch(hub_dir)
    if ep in reeval:
        if "FID" in filtered:
            filtered["FID"] = f"{reeval[ep]:.4f}"
        if "fid" in filtered:
            filtered["fid"] = f"{reeval[ep]:.4f}"
        if was_unverified:
            for q_key in ("Quality", "Quality_Score", "quality_score"):
                filtered.pop(q_key, None)
    elif was_unverified:
        for k in ("FID", "fid", "Quality", "Quality_Score", "quality_score"):
            filtered.pop(k, None)
    return filtered
