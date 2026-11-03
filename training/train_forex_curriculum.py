"""Forex walk-forward curriculum orchestrator: runs train.py fold by fold for forex_predictor."""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import yaml

from training.cli_args import add_clean_argument, add_timeframes_argument

logger = logging.getLogger("lemtrain.forex_curriculum")

NUM_FOLDS = 6
MODEL_KEY = "forex_predictor"
# Pair folders whose presence marks a directory as a forex manifold root.
KNOWN_PAIR_ANCHORS = {"EURUSD", "GBPUSD", "USDJPY", "XAUUSD"}


def get_latest_epoch(model_key, project_root):
    """Dynamically reads the latest epoch from metrics.csv."""
    metrics_path = os.path.abspath(
        os.path.join(project_root, "..", "LemGendaryModels", model_key, "metrics.csv")
    )
    if not os.path.exists(metrics_path):
        return 0
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if len(reader) > 0:
                return int(reader[-1].get("Epoch", 0))
    except (OSError, ValueError, TypeError, csv.Error) as e:
        logger.warning(
            "Failed to parse %s for dynamic target; assuming epoch 0: %s", metrics_path, e
        )
    return 0


@dataclass(frozen=True)
class _CurriculumPaths:
    """Filesystem locations used by the orchestrator."""

    project_root: str
    train_script: str
    ckpt_dir: str
    state_file: str


@dataclass
class _Curriculum:
    """One orchestrator run: CLI args, paths, persisted fold state and the fold plan."""

    args: argparse.Namespace
    paths: _CurriculumPaths
    state: dict
    active_folds: list

    def save_state(self) -> None:
        """Persist the fold state to ``curriculum_state.json``."""
        with open(self.paths.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    """Build the orchestrator's command-line parser."""
    parser = argparse.ArgumentParser(
        description="LemGendary Forex Walk-Forward Curriculum Orchestrator"
    )
    add_clean_argument(
        parser, "Start curriculum fresh from fold 1 epoch 1, wiping all checkpoints and states"
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of active folds to train (e.g. 1 2 3 4 5 6)",
    )
    add_timeframes_argument(
        parser, "Force specific active timeframes in minutes (e.g. 60 240 1440)"
    )
    parser.add_argument(
        "--epochs-per-fold",
        type=int,
        default=200,
        help="Max base epochs per fold before early stopping checks",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=str,
        default=None,
        help="Batch size per step (positive integer or 'auto')",
    )
    return parser


def _forex_search_dirs(project_root: str) -> list[str]:
    """Return the dataset directories scanned for forex manifolds."""
    base_search_dirs = [
        os.path.abspath(os.path.join(project_root, "..", "LemGendaryDatasets")),
        os.path.abspath(os.path.join(project_root, "data")),
    ]
    for mount in (
        "/kaggle/input",
        "/kaggle/working/LemGendaryDatasets",
        "/content/drive/MyDrive/LemGendaryDatasets",
    ):
        if os.path.exists(mount):
            base_search_dirs.append(mount)
    return base_search_dirs


def _forex_root_of(candidate_base: str, known_roots: list[str]) -> str | None:
    """Return the forex manifold root inside ``candidate_base``, or None when it is not one."""
    forex_sub = os.path.join(candidate_base, "forex")
    if os.path.isdir(forex_sub) and forex_sub not in known_roots:
        return forex_sub
    try:
        children = set(os.listdir(candidate_base))
    except OSError as exc:
        logger.debug("Cannot list forex candidate dir %s: %s", candidate_base, exc)
        return None
    if KNOWN_PAIR_ANCHORS & children or any(d.startswith("ForexUniverse") for d in children):
        return candidate_base
    return None


def _discover_forex_roots(base_search_dirs: list[str]) -> list[str]:
    """Return the distinct forex manifold roots found under ``base_search_dirs``."""
    resolved_roots: list[str] = []
    for base in base_search_dirs:
        if not os.path.exists(base):
            continue
        try:
            top_level = os.listdir(base)
        except OSError as exc:
            logger.debug("Cannot list forex search dir %s: %s", base, exc)
            continue
        for entry in top_level:
            if "forex" not in entry.lower():
                continue
            candidate_base = os.path.join(base, entry)
            if not os.path.isdir(candidate_base):
                continue
            root = _forex_root_of(candidate_base, resolved_roots)
            if root is not None and root not in resolved_roots:
                resolved_roots.append(root)
    return resolved_roots


def _discover_year_dirs(resolved_roots: list[str]) -> set[str]:
    """Return the ``ForexUniverse*`` year manifold names reachable from the roots."""
    all_year_dirs: set[str] = set()
    for root in resolved_roots:
        if not os.path.exists(root):
            continue
        bname = os.path.basename(root)
        if bname.startswith("ForexUniverse") and os.path.isdir(root):
            all_year_dirs.add(bname)
        try:
            for d in os.listdir(root):
                if (
                    d.startswith("ForexUniverse")
                    and not d.endswith(".zip")
                    and os.path.isdir(os.path.join(root, d))
                ):
                    all_year_dirs.add(d)
        except OSError as exc:
            logger.debug("Cannot list forex root %s: %s", root, exc)

    if os.path.exists("/kaggle/input") and len(all_year_dirs) < 8:
        try:
            for _, dirs, _ in os.walk("/kaggle/input"):
                for d in dirs:
                    if d.startswith("ForexUniverse") and not d.endswith(".zip"):
                        all_year_dirs.add(d)
        except OSError as exc:
            logger.debug("Cannot walk /kaggle/input for forex year manifolds: %s", exc)
    return all_year_dirs


def _count_manifold_folds(resolved_roots: list[str]) -> dict[str, int]:
    """Return the ``fold_*`` count of each manifold root (read from its first pair)."""
    manifold_folds = {}
    for root in resolved_roots:
        if not os.path.exists(root):
            continue
        pairs = [
            d
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
        ]
        if pairs:
            first_pair = pairs[0]
            fold_dirs = [
                d for d in os.listdir(os.path.join(root, first_pair)) if d.startswith("fold_")
            ]
            manifold_folds[root] = len(fold_dirs)
    return manifold_folds


def _validate_fold_consistency(project_root: str) -> None:
    """Report the discovered walk-forward folds; exit when attached manifolds disagree."""
    resolved_roots = _discover_forex_roots(_forex_search_dirs(project_root))
    all_year_dirs = _discover_year_dirs(resolved_roots)
    if all_year_dirs:
        total_folds = max(1, len(all_year_dirs) - 2)
        print(
            f" [DISCOVERY] [FOREX] Identified {len(all_year_dirs)} year manifolds "
            f"({', '.join(sorted(all_year_dirs))}) -> {total_folds} Walk-Forward Folds."
        )
        return

    manifold_folds = _count_manifold_folds(resolved_roots)
    if manifold_folds and len(set(manifold_folds.values())) > 1:
        print("\n[ERROR] Fold Mismatch Detected across attached manifolds!")
        for m, count in manifold_folds.items():
            print(f" - {os.path.basename(os.path.dirname(m))}: {count} folds")
        print(
            "\n[REQUIRED ACTION] Please ensure all mounted manifolds have the "
            "exact same number of folds."
        )
        sys.exit(1)


def _purge_previous_run(ckpt_dir: str) -> None:
    """Delete checkpoints, state files and telemetry of the previous curriculum (``--clean``)."""
    print(
        " [CLEAN] [RESET] Fresh curriculum run requested. "
        "Wiping checkpoints, state file, and telemetry..."
    )
    target_root = os.path.dirname(ckpt_dir)
    purge_targets = (
        [
            os.path.join(target_root, "metrics.csv"),
            *[
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                if f.endswith((".pth", ".json", ".processing"))
            ],
        ]
        if os.path.exists(ckpt_dir)
        else [os.path.join(target_root, "metrics.csv")]
    )
    for artifact in purge_targets:
        if os.path.isfile(artifact):
            try:
                os.unlink(artifact)
            except OSError as purge_err:
                print(f" [PURGE] Note: could not delete {os.path.basename(artifact)}: {purge_err}")


def _load_state(state_file: str) -> dict:
    """Load the persisted fold state (empty when missing or unreadable)."""
    if not os.path.exists(state_file):
        return {}
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(
            "Curriculum state %s unreadable; starting with empty state: %s", state_file, e
        )
        return {}


def _resolve_active_folds(project_root: str, args: argparse.Namespace) -> list:
    """Return the folds to train: CLI ``--folds`` > registry ``active_folds`` > all folds."""
    active_folds = list(range(1, NUM_FOLDS + 1))
    yaml_path = os.path.abspath(os.path.join(project_root, "unified_models_v2.yaml"))
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
                curr = y.get(MODEL_KEY, {}).get("curriculum", {})
                if "active_folds" in curr and isinstance(curr["active_folds"], list):
                    active_folds = curr["active_folds"]
        except (OSError, yaml.YAMLError, AttributeError, TypeError) as e:
            logger.warning("Failed to parse %s; using default active folds: %s", yaml_path, e)

    if getattr(args, "folds", None):
        active_folds = [f for f in args.folds if 1 <= f <= NUM_FOLDS]
    return active_folds


def _skip_if_completed(cur: _Curriculum, fold: int) -> bool:
    """Return True (and mark the state) when ``fold`` already finished in an earlier run."""
    fold_key = f"fold_{fold}"
    ckpt_dir = cur.paths.ckpt_dir
    is_completed = (
        cur.state.get(fold_key, {}).get("completed", False)
        or cur.state.get(f"phase4_fold{fold}", {}).get("completed", False)
        or os.path.exists(os.path.join(ckpt_dir, f"{MODEL_KEY}_fold{fold}.pth"))
        or os.path.exists(os.path.join(ckpt_dir, f"{MODEL_KEY}_phase4_fold{fold}.pth"))
    )
    if not is_completed:
        return False
    print(f" [ORCHESTRATOR] Skipping Fold {fold} (Already completed)")
    if not cur.state.get(fold_key, {}).get("completed"):
        cur.state[fold_key] = {"completed": True, "target": 0}
        cur.save_state()
    return True


def _plan_fold_target(cur: _Curriculum, fold_key: str, current_epoch: int) -> int:
    """Record and return the epoch budget target of the fold about to run."""
    max_epochs_per_fold = getattr(cur.args, "epochs_per_fold", 200)
    if fold_key not in cur.state:
        target_epoch = current_epoch + max_epochs_per_fold
        cur.state[fold_key] = {"completed": False, "target": target_epoch, "start": current_epoch}
    else:
        # [FIX] If the fold crashed mid-run, extend target from CURRENT epoch (not the
        # original start) so a restart doesn't undershoot the intended budget.
        if not cur.state[fold_key].get("completed", False):
            target_epoch = current_epoch + max_epochs_per_fold
        else:
            target_epoch = cur.state[fold_key].get("target", current_epoch + max_epochs_per_fold)
        cur.state[fold_key]["target"] = target_epoch
    cur.save_state()
    return target_epoch


def _build_fold_command(cur: _Curriculum, fold: int, target_epoch: int) -> list[str]:
    """Return the ``train.py`` command line for one fold."""
    args = cur.args
    # 2026: --enable-batch-growth lets the governor raise physical batch
    # toward VRAM ceiling. Workers default to the env-based count (4 for
    # Kaggle) — the LRU row-group cache needs them to keep the GPU fed.
    cmd = [
        sys.executable,
        cur.paths.train_script,
        "--model",
        MODEL_KEY,
        "--epochs",
        str(target_epoch),
        "--fold",
        str(fold),
        "--enable-batch-growth",
    ]

    if getattr(args, "batch_size", None):
        cmd.extend(["--batch_size", str(args.batch_size)])

    if getattr(args, "timeframes", None):
        cmd.append("--timeframes")
        cmd.extend([str(t) for t in args.timeframes])

    if getattr(args, "clean", False) and fold == cur.active_folds[0]:
        cmd.append("--clean")

    env_type = (
        "kaggle"
        if os.path.exists("/kaggle")
        else ("colab" if os.path.exists("/content") else "local")
    )
    if env_type in ["kaggle", "colab"]:
        cmd.extend(["--env", env_type, "--auto_sync"])
    return cmd


def _execute_fold(cur: _Curriculum, fold: int, cmd: list[str]) -> None:
    """Run the fold's training subprocess and mark the fold completed; exit on failure."""
    fold_key = f"fold_{fold}"
    try:
        cflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if sys.platform == "win32" else 0
        )
        res = subprocess.run(cmd, check=False, creationflags=cflags)
        if res.returncode != 0:
            print(f"\n [ERROR] Training crashed during Fold {fold}. Exiting curriculum.")
            sys.exit(1)

        completed_epoch = get_latest_epoch(MODEL_KEY, cur.paths.project_root)
        print(f" [ORCHESTRATOR] Fold {fold} completed at Epoch {completed_epoch}.")

        cur.state[fold_key]["completed"] = True
        cur.state[fold_key]["finished_at"] = completed_epoch
        cur.save_state()

    except KeyboardInterrupt:
        print("\n [INTERRUPT] Caught KeyboardInterrupt. Exiting curriculum orchestrator safely.")
        sys.exit(0)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError) as e:
        logger.error("Fold %s subprocess or state persistence failed: %s", fold, e)
        sys.exit(1)


def _archive_fold_checkpoint(ckpt_dir: str, fold: int) -> None:
    """Copy the latest checkpoint to the fold archive for historical preservation."""
    archive_ckpt = os.path.join(ckpt_dir, f"{MODEL_KEY}_fold{fold}.pth")
    latest_ckpt = os.path.join(ckpt_dir, f"{MODEL_KEY}_latest.pth")
    if os.path.exists(latest_ckpt):
        try:
            shutil.copy2(latest_ckpt, archive_ckpt)
            print(f" [ARCHIVE] Preserved state: {archive_ckpt}")
        except OSError as e:
            logger.warning(
                "Failed to archive checkpoint %s -> %s: %s", latest_ckpt, archive_ckpt, e
            )


def _run_fold(cur: _Curriculum, fold: int) -> None:
    """Train one walk-forward fold unless it is already complete."""
    if _skip_if_completed(cur, fold):
        return

    current_epoch = get_latest_epoch(MODEL_KEY, cur.paths.project_root)
    target_epoch = _plan_fold_target(cur, f"fold_{fold}", current_epoch)

    print("\n================================================================================")
    print(f"--- Launching Walk-Forward Fold {fold}/{NUM_FOLDS} ---")
    print(f" [ORCHESTRATOR] Current Epoch: {current_epoch} | Max Target Epoch: {target_epoch}")
    print("================================================================================\n")

    cmd = _build_fold_command(cur, fold, target_epoch)
    print(f" [EXEC] {' '.join(cmd)}")
    _execute_fold(cur, fold, cmd)
    _archive_fold_checkpoint(cur.paths.ckpt_dir, fold)
    time.sleep(2)


def main():
    """Parse CLI args and run the walk-forward fold curriculum, resuming from saved state."""
    args = _build_parser().parse_args()

    print("================================================================================")
    print("  LEMGENDARY FOREX WALK-FORWARD CURRICULUM ORCHESTRATOR")
    print("  Executing Walk-Forward Matrix across 16-Symbol Universe (Folds 1 -> 6)")
    print("================================================================================\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 2026: Validate fold consistency across attached manifolds
    _validate_fold_consistency(project_root)

    ckpt_dir = os.path.abspath(
        os.path.join(project_root, "..", "LemGendaryModels", MODEL_KEY, "checkpoints")
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    if getattr(args, "clean", False):
        _purge_previous_run(ckpt_dir)

    state_file = os.path.join(ckpt_dir, "curriculum_state.json")
    cur = _Curriculum(
        args=args,
        paths=_CurriculumPaths(
            project_root=project_root,
            train_script=os.path.join(script_dir, "train.py"),
            ckpt_dir=ckpt_dir,
            state_file=state_file,
        ),
        state=_load_state(state_file),
        active_folds=_resolve_active_folds(project_root, args),
    )

    print(f" [ORCHESTRATOR] Active Walk-Forward Folds: {cur.active_folds}")

    for fold in cur.active_folds:
        _run_fold(cur, fold)

    print("\n================================================================================")
    print(" [SUCCESS] FULL FOREX WALK-FORWARD CURRICULUM COMPLETED.")
    print("================================================================================")


if __name__ == "__main__":
    main()
