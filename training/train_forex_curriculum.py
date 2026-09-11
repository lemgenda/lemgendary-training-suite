import os
import sys
import subprocess
import time
import shutil
import csv
import json
import argparse
import yaml

NUM_FOLDS = 6
MODEL_KEY = "forex_predictor"

def get_latest_epoch(model_key, project_root):
    """Dynamically reads the latest epoch from metrics.csv."""
    metrics_path = os.path.abspath(os.path.join(project_root, "..", "LemGendaryModels", model_key, "metrics.csv"))
    if not os.path.exists(metrics_path):
        return 0
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if len(reader) > 0:
                return int(reader[-1].get("Epoch", 0))
    except Exception as e:
        print(f" [WARNING] Failed to parse metrics.csv for dynamic target: {e}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="LemGendary Forex Walk-Forward Curriculum Orchestrator")
    parser.add_argument("--clean", "--fresh", dest="clean", action="store_true", help="Start curriculum fresh from fold 1 epoch 1, wiping all checkpoints and states")
    parser.add_argument("--folds", type=int, nargs='+', default=None, help="Explicit list of active folds to train (e.g. 1 2 3 4 5 6)")
    parser.add_argument("--timeframes", type=int, nargs='+', default=None, help="Force specific active timeframes in minutes (e.g. 60 240 1440)")
    parser.add_argument("--epochs-per-fold", type=int, default=200, help="Max base epochs per fold before early stopping checks")
    args = parser.parse_args()

    print("================================================================================")
    print("  LEMGENDARY FOREX WALK-FORWARD CURRICULUM ORCHESTRATOR")
    print("  Executing Walk-Forward Matrix across 16-Symbol Universe (Folds 1 -> 6)")
    print("================================================================================\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    project_root = os.path.dirname(script_dir)

    # 2026: Validate fold consistency across attached manifolds
    KNOWN_PAIR_ANCHORS = {"EURUSD", "GBPUSD", "USDJPY", "XAUUSD"}
    resolved_roots = []
    base_search_dirs = [
        os.path.abspath(os.path.join(project_root, "..", "LemGendaryDatasets")),
        os.path.abspath(os.path.join(project_root, "data"))
    ]
    if os.path.exists("/kaggle/input"):
        base_search_dirs.append("/kaggle/input")
    if os.path.exists("/kaggle/working/LemGendaryDatasets"):
        base_search_dirs.append("/kaggle/working/LemGendaryDatasets")
    if os.path.exists("/content/drive/MyDrive/LemGendaryDatasets"):
        base_search_dirs.append("/content/drive/MyDrive/LemGendaryDatasets")

    for base in base_search_dirs:
        if not os.path.exists(base):
            continue
        try:
            top_level = os.listdir(base)
        except OSError:
            continue
        for entry in top_level:
            if "forex" not in entry.lower():
                continue
            candidate_base = os.path.join(base, entry)
            if not os.path.isdir(candidate_base):
                continue
            forex_sub = os.path.join(candidate_base, "forex")
            if os.path.isdir(forex_sub) and forex_sub not in resolved_roots:
                resolved_roots.append(forex_sub)
                continue
            try:
                children = set(os.listdir(candidate_base))
            except OSError:
                continue
            if KNOWN_PAIR_ANCHORS & children:
                if candidate_base not in resolved_roots:
                    resolved_roots.append(candidate_base)
                continue
            if any(d.startswith("ForexUniverse") for d in children):
                if candidate_base not in resolved_roots:
                    resolved_roots.append(candidate_base)
                continue

    all_year_dirs = set()
    for root in resolved_roots:
        if not os.path.exists(root):
            continue
        bname = os.path.basename(root)
        if bname.startswith("ForexUniverse") and os.path.isdir(root):
            all_year_dirs.add(bname)
        try:
            for d in os.listdir(root):
                if d.startswith("ForexUniverse") and not d.endswith(".zip") and os.path.isdir(os.path.join(root, d)):
                    all_year_dirs.add(d)
        except OSError:
            pass

    if os.path.exists('/kaggle/input') and len(all_year_dirs) < 8:
        try:
            for root_dir, dirs, _ in os.walk('/kaggle/input'):
                for d in dirs:
                    if d.startswith("ForexUniverse") and not d.endswith(".zip"):
                        all_year_dirs.add(d)
        except OSError:
            pass

    if all_year_dirs:
        total_folds = max(1, len(all_year_dirs) - 2)
        print(f" [DISCOVERY] [FOREX] Identified {len(all_year_dirs)} year manifolds ({', '.join(sorted(all_year_dirs))}) -> {total_folds} Walk-Forward Folds.")
    else:
        manifold_folds = {}
        for root in resolved_roots:
            if not os.path.exists(root):
                continue
            pairs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')]
            if pairs:
                first_pair = pairs[0]
                fold_dirs = [d for d in os.listdir(os.path.join(root, first_pair)) if d.startswith("fold_")]
                manifold_folds[root] = len(fold_dirs)

        if manifold_folds:
            unique_fold_counts = set(manifold_folds.values())
            if len(unique_fold_counts) > 1:
                print("\n[ERROR] Fold Mismatch Detected across attached manifolds!")
                for m, count in manifold_folds.items():
                    print(f" - {os.path.basename(os.path.dirname(m))}: {count} folds")
                print("\n[REQUIRED ACTION] Please ensure all mounted manifolds have the exact same number of folds.")
                sys.exit(1)
            
    ckpt_dir = os.path.abspath(os.path.join(project_root, "..", "LemGendaryModels", MODEL_KEY, "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)

    if getattr(args, 'clean', False):
        print(" [CLEAN] [RESET] Fresh curriculum run requested. Wiping checkpoints, state file, and telemetry...")
        target_root = os.path.dirname(ckpt_dir)
        purge_targets = [
            os.path.join(target_root, "metrics.csv"),
            *[os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith((".pth", ".json", ".processing"))]
        ] if os.path.exists(ckpt_dir) else [os.path.join(target_root, "metrics.csv")]
        for artifact in purge_targets:
            if os.path.isfile(artifact):
                try:
                    os.unlink(artifact)
                except OSError as purge_err:
                    print(f" [PURGE] Note: could not delete {os.path.basename(artifact)}: {purge_err}")

    state_file = os.path.join(ckpt_dir, "curriculum_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
    else:
        state = {}

    active_folds = list(range(1, NUM_FOLDS + 1))
    yaml_path = os.path.abspath(os.path.join(project_root, "unified_models_v2.yaml"))
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
                curr = y.get(MODEL_KEY, {}).get("curriculum", {})
                if "active_folds" in curr and isinstance(curr["active_folds"], list):
                    active_folds = curr["active_folds"]
        except Exception as e:
            print(f" [WARNING] Failed to parse unified_models_v2.yaml: {e}")

    if getattr(args, 'folds', None):
        active_folds = [f for f in args.folds if 1 <= f <= NUM_FOLDS]

    print(f" [ORCHESTRATOR] Active Walk-Forward Folds: {active_folds}")
    max_epochs_per_fold = getattr(args, 'epochs_per_fold', 200)

    for fold in active_folds:
        fold_key = f"fold_{fold}"
        legacy_fold_key = f"phase4_fold{fold}"
        archive_ckpt = os.path.join(ckpt_dir, f"{MODEL_KEY}_fold{fold}.pth")
        legacy_archive_ckpt = os.path.join(ckpt_dir, f"{MODEL_KEY}_phase4_fold{fold}.pth")
        
        # Check if fold is already complete
        is_completed = (
            state.get(fold_key, {}).get("completed", False)
            or state.get(legacy_fold_key, {}).get("completed", False)
            or os.path.exists(archive_ckpt)
            or os.path.exists(legacy_archive_ckpt)
        )
        if is_completed:
            print(f" [ORCHESTRATOR] Skipping Fold {fold} (Already completed)")
            if not state.get(fold_key, {}).get("completed"):
                state[fold_key] = {"completed": True, "target": 0}
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
            continue

        current_epoch = get_latest_epoch(MODEL_KEY, project_root)
        
        if fold_key not in state:
            target_epoch = current_epoch + max_epochs_per_fold
            state[fold_key] = {"completed": False, "target": target_epoch, "start": current_epoch}
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        else:
            fold_start = state[fold_key].get("start", current_epoch)
            target_epoch = fold_start + max_epochs_per_fold
            state[fold_key]["target"] = target_epoch
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

        print(f"\n================================================================================")
        print(f"--- Launching Walk-Forward Fold {fold}/{NUM_FOLDS} ---")
        print(f" [ORCHESTRATOR] Current Epoch: {current_epoch} | Max Target Epoch: {target_epoch}")
        print(f"================================================================================\n")
        
        cmd = [
            sys.executable, train_script,
            "--model", MODEL_KEY,
            "--epochs", str(target_epoch),
            "--fold", str(fold)
        ]
        
        if getattr(args, 'timeframes', None):
            cmd.append("--timeframes")
            cmd.extend([str(t) for t in args.timeframes])

        if getattr(args, 'clean', False) and fold == active_folds[0]:
            cmd.append("--clean")
        
        env_type = "kaggle" if os.path.exists("/kaggle") else ("colab" if os.path.exists("/content") else "local")
        if env_type in ["kaggle", "colab"]:
            cmd.extend(["--env", env_type, "--auto_sync"])
        
        print(f" [EXEC] {' '.join(cmd)}")
        
        try:
            res = subprocess.run(cmd, check=False)
            if res.returncode != 0:
                print(f"\n [ERROR] Training crashed during Fold {fold}. Exiting curriculum.")
                sys.exit(1)
            
            completed_epoch = get_latest_epoch(MODEL_KEY, project_root)
            print(f" [ORCHESTRATOR] Fold {fold} completed at Epoch {completed_epoch}.")
            
            state[fold_key]["completed"] = True
            state[fold_key]["finished_at"] = completed_epoch
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            
        except KeyboardInterrupt:
            print("\n [INTERRUPT] Caught KeyboardInterrupt. Exiting curriculum orchestrator safely.")
            sys.exit(0)
        except Exception as e:
            print(f"\n [ERROR] Subprocess failed: {e}")
            sys.exit(1)
                
        # Copy checkpoint state at the end of the fold for historical preservation
        latest_ckpt = os.path.join(ckpt_dir, f"{MODEL_KEY}_latest.pth")
        if os.path.exists(latest_ckpt):
            try:
                shutil.copy2(latest_ckpt, archive_ckpt)
                print(f" [ARCHIVE] Preserved state: {archive_ckpt}")
            except Exception as e:
                print(f" [WARNING] Failed to archive checkpoint: {e}")
                
        time.sleep(2)

    print("\n================================================================================")
    print(" [SUCCESS] FULL FOREX WALK-FORWARD CURRICULUM COMPLETED.")
    print("================================================================================")

if __name__ == "__main__":
    main()
