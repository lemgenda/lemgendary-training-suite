import os
import sys
import subprocess
import time
import shutil
import csv
import argparse

# --- Walk-Forward Curriculum Configuration ---
CURRICULUM_PHASES = [
    {
        "phase": 1,
        "name": "Titan 4 Core",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
        "max_epochs": 200 # Dynamic ceiling; early stopping dictates actual length
    },
    {
        "phase": 2,
        "name": "G7 Majors (8)",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCAD", "USDCHF", "AUDUSD", "NZDUSD"],
        "max_epochs": 200
    },
    {
        "phase": 3,
        "name": "High-Beta Crosses (12)",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCAD", "USDCHF", "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "USOIL"],
        "max_epochs": 200
    },
    {
        "phase": 4,
        "name": "Full Universe (16)",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCAD", "USDCHF", "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "USOIL", "US500", "USTEC", "GER40", "XAGUSD"],
        "max_epochs": 200
    }
]

NUM_FOLDS = 6
MODEL_KEY = "forex_predictor"

def get_latest_epoch(model_key, project_root):
    """Dynamically reads the latest epoch from metrics.csv."""
    metrics_path = os.path.abspath(os.path.join(project_root, "..", "LemGendaryModels", model_key, "metrics.csv"))
    if not os.path.exists(metrics_path):
        return 0
    try:
        import csv
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if len(reader) > 0:
                return int(reader[-1].get("Epoch", 0))
    except Exception as e:
        print(f" [WARNING] Failed to parse metrics.csv for dynamic target: {e}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="LemGendary Forex Curriculum Orchestrator")
    parser.add_argument("--clean", "--fresh", dest="clean", action="store_true", help="Start curriculum fresh from fold 1 epoch 1, wiping all checkpoints and states")
    args = parser.parse_args()

    print("================================================================================")
    print("  LEMGENDARY FOREX CURRICULUM ORCHESTRATOR")
    print("  Executing Walk-Forward Expansion Matrix (4 -> 16 Pairs | Folds 1 -> 6)")
    print("================================================================================\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    
    # 2026: Auto-resolve checkpoint dir to preserve phase states
    project_root = os.path.dirname(script_dir)
    
    # 2026: Validate multiphase fold consistency across attached manifolds
    # Targeted search: only scan directories whose names contain 'forex'/'Forex'
    # to avoid an expensive full os.walk across all large image dataset directories.
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
            # Only descend into directories that are forex-related by name
            if "forex" not in entry.lower():
                continue
            candidate_base = os.path.join(base, entry)
            if not os.path.isdir(candidate_base):
                continue
            # Pattern 1: candidate_base/forex/<PAIR>/fold_N
            forex_sub = os.path.join(candidate_base, "forex")
            if os.path.isdir(forex_sub) and forex_sub not in resolved_roots:
                resolved_roots.append(forex_sub)
                continue
            # Pattern 2: candidate_base/<PAIR>/fold_N (pairs directly inside)
            try:
                children = set(os.listdir(candidate_base))
            except OSError:
                continue
            if KNOWN_PAIR_ANCHORS & children:
                if candidate_base not in resolved_roots:
                    resolved_roots.append(candidate_base)

    manifold_folds = {}
    for root in resolved_roots:
        if not os.path.exists(root): continue
        pairs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')]
        if pairs:
            first_pair = pairs[0]
            fold_dirs = [d for d in os.listdir(os.path.join(root, first_pair)) if d.startswith("fold_")]
            manifold_folds[root] = len(fold_dirs)

    if manifold_folds:
        unique_fold_counts = set(manifold_folds.values())
        if len(unique_fold_counts) > 1:
            print(f"\n[ERROR] Multiphase Fold Mismatch Detected!")
            print(f"Attached manifolds have different number of folds:")
            for m, count in manifold_folds.items():
                print(f" - {os.path.basename(os.path.dirname(m))}: {count} folds")
            print("\n[REQUIRED ACTION] You are using reduced manifolds with a mismatched number of folds.")
            print("Please ensure all mounted manifolds have the exact same number of folds before continuing.")
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
    import json
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
    else:
        state = {}

    import yaml
    active_phases = [1, 2, 3, 4]
    active_folds = list(range(1, NUM_FOLDS + 1))
    yaml_path = os.path.abspath(os.path.join(project_root, "unified_models_v2.yaml"))
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
                curr = y.get(MODEL_KEY, {}).get("curriculum", {})
                if "active_phases" in curr: active_phases = curr["active_phases"]
                if "active_folds" in curr: active_folds = curr["active_folds"]
        except Exception as e:
            print(f" [WARNING] Failed to parse unified_models_v2.yaml: {e}")

    for phase in CURRICULUM_PHASES:
        p_id = phase["phase"]
        if p_id not in active_phases:
            continue

        p_name = phase["name"]
        pairs = phase["pairs"]
        max_epochs_raw = phase["max_epochs"]
        assert isinstance(pairs, list)
        assert isinstance(max_epochs_raw, (int, str))
        max_epochs_per_fold = int(max_epochs_raw)
        
        print(f"\n>>>>> ENTERING PHASE {p_id}: {p_name} <<<<<")
        print(f"      Pairs: {len(pairs)} active")
        
        for fold in range(1, NUM_FOLDS + 1):
            if fold not in active_folds:
                continue
            fold_key = f"phase{p_id}_fold{fold}"
            archive_ckpt = os.path.join(ckpt_dir, f"{MODEL_KEY}_phase{p_id}_fold{fold}.pth")
            
            # Check if fold is already complete
            if state.get(fold_key, {}).get("completed", False) or os.path.exists(archive_ckpt):
                print(f" [ORCHESTRATOR] Skipping {fold_key} (Already completed)")
                # Self-heal state file if only archive exists
                if not state.get(fold_key, {}).get("completed"):
                    state[fold_key] = {"completed": True, "target": 0}
                    with open(state_file, "w") as f: json.dump(state, f)
                continue

            current_epoch = get_latest_epoch(MODEL_KEY, project_root)
            
            # Assign and lock target for this fold if it hasn't been started
            if fold_key not in state:
                target_epoch = current_epoch + max_epochs_per_fold
                state[fold_key] = {"completed": False, "target": target_epoch, "start": current_epoch}
                with open(state_file, "w") as f: json.dump(state, f)
            else:
                # 2026 Dynamic Walk-Forward: Ensure we respect the original start epoch for max_epochs calculation
                fold_start = state[fold_key].get("start", current_epoch)
                target_epoch = fold_start + max_epochs_per_fold
                state[fold_key]["target"] = target_epoch
                with open(state_file, "w") as f: json.dump(state, f)

            print(f"\n--- Launching Phase {p_id} | Fold {fold}/{NUM_FOLDS} ---")
            print(f" [ORCHESTRATOR] Current Epoch: {current_epoch} | Max Target Epoch: {target_epoch}")
            
            # Construct command
            cmd = [
                sys.executable, train_script,
                "--model", MODEL_KEY,
                "--epochs", str(target_epoch),
                "--fold", str(fold)
            ]
            
            # Add pairs argument
            cmd.append("--pairs")
            cmd.extend(pairs)

            if getattr(args, 'clean', False) and p_id == 1 and fold == 1:
                cmd.append("--clean")
            
            # 2026 Cloud Environment Auto-Forwarding
            env_type = "kaggle" if os.path.exists("/kaggle") else ("colab" if os.path.exists("/content") else "local")
            if env_type in ["kaggle", "colab"]:
                cmd.extend(["--env", env_type, "--auto_sync"])
            
            print(f" [EXEC] {' '.join(cmd)}")
            
            # 2026 Resilience: Run subprocess, block until complete. 
            try:
                res = subprocess.run(cmd)
                if res.returncode != 0:
                    print(f"\n [ERROR] Training crashed during Phase {p_id} Fold {fold}. Exiting curriculum.")
                    sys.exit(1)
                
                # Check actual completed epoch (dynamic due to early stopping)
                completed_epoch = get_latest_epoch(MODEL_KEY, project_root)
                print(f" [ORCHESTRATOR] Fold {fold} completed at Epoch {completed_epoch}.")
                
                # Update state
                state[fold_key]["completed"] = True
                state[fold_key]["finished_at"] = completed_epoch
                with open(state_file, "w") as f: json.dump(state, f)
                
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
                    
            # Mark fold as completed
            state[fold_key]["completed"] = True
            with open(state_file, "w") as f: json.dump(state, f)
                    
        print(f"\n<<<<< PHASE {p_id} ({p_name}) COMPLETE >>>>>")
        time.sleep(2)

    print("\n================================================================================")
    print(" [SUCCESS] FULL FOREX CURRICULUM WALK-FORWARD MATRIX COMPLETED.")
    print("================================================================================")

if __name__ == "__main__":
    main()
