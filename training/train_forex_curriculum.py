import os
import sys
import subprocess
import time
import shutil
import csv

# --- Walk-Forward Curriculum Configuration ---
CURRICULUM_PHASES = [
    {
        "phase": 1,
        "name": "Titan 4 Core",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
        "epochs": 50 # Base epochs per fold in this phase
    },
    {
        "phase": 2,
        "name": "G7 Majors (8)",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCAD", "USDCHF", "AUDUSD", "NZDUSD"],
        "epochs": 40
    },
    {
        "phase": 3,
        "name": "High-Beta Crosses (12)",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCAD", "USDCHF", "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "USOIL"],
        "epochs": 30
    },
    {
        "phase": 4,
        "name": "Full Universe (16)",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDCAD", "USDCHF", "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "USOIL", "US500", "USTEC", "GER40", "XAGUSD"],
        "epochs": 20
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
        with open(metrics_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            if len(lines) > 1 and lines[-1].strip():
                return int(lines[-1].split(',')[0])
    except Exception as e:
        print(f" [WARNING] Failed to parse metrics.csv for dynamic target: {e}")
    return 0

def main():
    print("================================================================================")
    print("  LEMGENDARY FOREX CURRICULUM ORCHESTRATOR")
    print("  Executing Walk-Forward Expansion Matrix (4 -> 16 Pairs | Folds 1 -> 6)")
    print("================================================================================\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    
    # 2026: Auto-resolve checkpoint dir to preserve phase states
    project_root = os.path.dirname(script_dir)
    ckpt_dir = os.path.abspath(os.path.join(project_root, "..", "LemGendaryModels", MODEL_KEY, "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)

    for phase in CURRICULUM_PHASES:
        p_id = phase["phase"]
        p_name = phase["name"]
        pairs = phase["pairs"]
        epochs_raw = phase["epochs"]
        assert isinstance(pairs, list)
        assert isinstance(epochs_raw, (int, str))
        epochs_per_fold = int(epochs_raw)
        
        print(f"\n>>>>> ENTERING PHASE {p_id}: {p_name} <<<<<")
        print(f"      Pairs: {len(pairs)} active")
        
        for fold in range(1, NUM_FOLDS + 1):
            print(f"\n--- Launching Phase {p_id} | Fold {fold}/{NUM_FOLDS} ---")
            
            # Dynamic Epoch Scaling: Read actual current epoch to prevent resiliency guardrail starvation
            current_epoch = get_latest_epoch(MODEL_KEY, project_root)
            global_target_epochs = current_epoch + epochs_per_fold
            print(f" [ORCHESTRATOR] Current Epoch: {current_epoch} | Target Epoch: {global_target_epochs}")
            
            # Construct command
            cmd = [
                sys.executable, train_script,
                "--model", MODEL_KEY,
                "--epochs", str(global_target_epochs),
                "--fold", str(fold)
            ]
            
            # Add pairs argument
            cmd.append("--pairs")
            cmd.extend(pairs)
            
            print(f" [EXEC] {' '.join(cmd)}")
            
            # 2026 Resilience: Run subprocess, block until complete. 
            try:
                res = subprocess.run(cmd)
                if res.returncode != 0:
                    print(f"\n[CRITICAL ERROR] Training aborted in Phase {p_id}, Fold {fold}. Orchestrator halted.")
                    sys.exit(res.returncode)
            except KeyboardInterrupt:
                print("\n[INTERRUPT] Curriculum Orchestrator halted by user.")
                sys.exit(0)
                
            # Copy checkpoint state at the end of the fold for historical preservation
            latest_ckpt = os.path.join(ckpt_dir, f"{MODEL_KEY}_latest.pth")
            archive_ckpt = os.path.join(ckpt_dir, f"{MODEL_KEY}_phase{p_id}_fold{fold}.pth")
            if os.path.exists(latest_ckpt):
                try:
                    shutil.copy2(latest_ckpt, archive_ckpt)
                    print(f" [ARCHIVE] Preserved state: {archive_ckpt}")
                except Exception as e:
                    print(f" [WARNING] Failed to archive checkpoint: {e}")
                    
        print(f"\n<<<<< PHASE {p_id} ({p_name}) COMPLETE >>>>>")
        time.sleep(2)

    print("\n================================================================================")
    print(" [SUCCESS] FULL FOREX CURRICULUM WALK-FORWARD MATRIX COMPLETED.")
    print("================================================================================")

if __name__ == "__main__":
    main()
