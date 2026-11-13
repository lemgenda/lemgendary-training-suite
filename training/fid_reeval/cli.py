"""Command line of the FID re-evaluation tool: ``python -m training.fid_reeval``.

Subcommands:

* ``measure``: real FID of one checkpoint on the model's validation split (``--dry-run`` checks
  every input on CPU without computing FID);
* ``mark-unverified``: record the formula-rewritten FID rows of the four affected models as
  ``unverified`` (idempotent);
* ``status``: per-model summary of the ledger.

All results are appended to ``<models-root>/<model>/fid_provenance.csv``. ``metrics.csv``,
checkpoints and governor/rung state are never written.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.fid_reeval import ledger
from training.fid_reeval.inputs import InputError, ReevalRequest, ResolvedInputs, resolve_inputs
from training.fid_reeval.measure import (
    FID_FEATURE_DIM,
    FidInvariantError,
    FidMeasurement,
    FidUnavailableError,
    fid_backend_available,
    measure_fid,
)
from training.utils.paths import get_models_hub_root, get_project_root

EXIT_INPUT = 2
EXIT_FID = 3


def _models_root(args: argparse.Namespace) -> Path:
    return Path(args.models_root) if args.models_root else get_models_hub_root()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models-root", help="LemGendaryModels directory (default: <workspace>/LemGendaryModels)"
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")


def _add_selection(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=ledger.FABRICATED_FID_MODELS)
    group.add_argument("--all", action="store_true", help="all four affected models")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with the ``measure``, ``mark-unverified`` and ``status`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m training.fid_reeval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="measure real FID of a checkpoint")
    measure.add_argument("--model", required=True, help="model id from unified_models_v2.yaml")
    measure.add_argument("--checkpoint", help="default: <models-root>/<model>/checkpoints/best.pth")
    measure.add_argument("--datasets-root", help="folder holding the model's manifold folders")
    measure.add_argument("--env", choices=("local", "kaggle", "colab"), default="local")
    measure.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    measure.add_argument("--batch-size", type=int, default=4)
    measure.add_argument("--num-workers", type=int, default=0)
    measure.add_argument("--non-strict", action="store_true", help="allow missing/extra keys")
    measure.add_argument("--note", default="", help="free text stored in the ledger row")
    _add_common(measure)

    mark = sub.add_parser("mark-unverified", help="mark recalibrated FID rows as unverified")
    _add_selection(mark)
    _add_common(mark)

    status = sub.add_parser("status", help="summarise the FID provenance ledgers")
    _add_selection(status)
    status.add_argument("--models-root")
    return parser


def _selected(args: argparse.Namespace) -> tuple[str, ...]:
    return ledger.FABRICATED_FID_MODELS if args.all else (args.model,)


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise InputError(f"--device {name} requested but CUDA is not available")
    if name == "cpu":
        print("[FID] WARNING: running on CPU; expect hours for a full validation split.")
    return torch.device(name)


def _request(args: argparse.Namespace) -> ReevalRequest:
    if args.batch_size < 1:
        raise InputError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise InputError("--num-workers must not be negative")
    hub = _models_root(args) / args.model
    checkpoint = Path(args.checkpoint) if args.checkpoint else hub / "checkpoints" / "best.pth"
    return ReevalRequest(
        model_id=args.model,
        checkpoint=checkpoint,
        datasets_root=Path(args.datasets_root) if args.datasets_root else None,
        env=args.env,
        strict=not args.non_strict,
    )


def _ledger_row(
    resolved: ResolvedInputs, result: FidMeasurement, device: str, note: str
) -> ledger.LedgerRow:
    req = resolved.request
    manifolds = "|".join(sorted(getattr(resolved.dataset, "path_cache", {}) or {}))
    notes = [resolved.load_report, f"batches={result.batches}", f"manifolds={manifolds}"]
    if result.n_images < FID_FEATURE_DIM:
        notes.append(f"n_images<{FID_FEATURE_DIM}: FID biased upward, compare like-for-like only")
    if note:
        notes.append(note)
    root = req.datasets_root or resolved.config.get("paths", {}).get("datasets_root", "")
    return ledger.new_row(
        recorded_at=ledger.utc_now(),
        model_id=req.model_id,
        status=ledger.STATUS_REEVALUATED,
        source=ledger.SOURCE_REEVALUATED,
        epoch=resolved.checkpoint_epoch or "unknown",
        metric="FID",
        value=f"{result.fid:.4f}",
        n_images=str(result.n_images),
        feature_dim=str(result.feature_dim),
        max_chunk=str(result.max_chunk),
        checkpoint=str(req.checkpoint.resolve()),
        checkpoint_sha256=resolved.checkpoint_sha256,
        datasets_root=str(root),
        device=device,
        note="; ".join(notes),
    )


def cmd_measure(args: argparse.Namespace) -> int:
    """``measure``: validate inputs, then (unless ``--dry-run``) compute and record FID."""
    request = _request(args)
    resolved = resolve_inputs(request, get_project_root())
    print(
        f"[FID] {request.model_id}: checkpoint {request.checkpoint} (epoch "
        f"{resolved.checkpoint_epoch or '?'}, sha256 {resolved.checkpoint_sha256[:12]}), "
        f"{resolved.load_report}, {len(resolved.dataset)} validation images"
    )
    if not fid_backend_available():
        raise FidUnavailableError("torchmetrics FID is not importable; install requirements.txt")
    device = _device(args.device)
    if args.dry_run:
        print("[FID] dry run: inputs valid, nothing measured or written.")
        return 0
    loader = DataLoader(
        resolved.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    result = measure_fid(resolved.model.to(device), loader, device)
    row = _ledger_row(resolved, result, str(device), args.note)
    path = ledger.ledger_path(_models_root(args) / request.model_id)
    # Print first: the number survives even if the ledger append is refused.
    print(f"[FID] {request.model_id}: FID={row['value']} over {result.n_images} images")
    ledger.append_rows(path, [row])
    print(f"[FID] appended to {path}")
    return 0


def cmd_mark_unverified(args: argparse.Namespace) -> int:
    """``mark-unverified``: append ``unverified`` rows for rewritten FID epochs."""
    for model_id in _selected(args):
        hub = _models_root(args) / model_id
        if not (hub / ledger.METRICS_NAME).exists():
            print(f"[FID] {model_id}: no {hub / ledger.METRICS_NAME}, skipped")
            continue
        rows = ledger.plan_unverified(hub, model_id)
        verb = "would mark" if args.dry_run else "marked"
        if not args.dry_run:
            ledger.append_rows(ledger.ledger_path(hub), rows)
        print(f"[FID] {model_id}: {verb} {len(rows)} epoch(s) unverified")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """``status``: one line per model."""
    for model_id in _selected(args):
        st = ledger.ledger_status(_models_root(args) / model_id, model_id)
        latest = (
            f"latest FID {st.latest['value']} (epoch {st.latest['epoch']}, "
            f"{st.latest['recorded_at']})"
            if st.latest
            else "never re-measured"
        )
        print(
            f"[FID] {model_id}: {st.unverified_epochs} unverified epoch(s), "
            f"{st.reevaluations} re-evaluation(s), {latest}"
        )
    return 0


_COMMANDS = {"measure": cmd_measure, "mark-unverified": cmd_mark_unverified, "status": cmd_status}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (InputError, ledger.LedgerError) as exc:
        print(f"[FID] ERROR: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except (FidInvariantError, FidUnavailableError) as exc:
        print(f"[FID] FAILED: {exc}", file=sys.stderr)
        return EXIT_FID
