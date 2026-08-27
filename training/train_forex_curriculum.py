import os
import sys
import subprocess
import time
import shutil

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

def main():
    print("================================================================================")
    print("  LEMGENDARY FOREX CURRICULUM ORCHESTRATOR")
    print("  Executing Walk-Forward Expansion Matrix (4 -> 16 Pairs | Folds 1 -> 6)")
    print("================================================================================\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    
    # 2026: Auto-resolve checkpoint dir to preserve phase states
    project_root = os.path.dirname(script_dir)
    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    global_target_epochs = 0

    for phase in CURRICULUM_PHASES:
        p_id = phase["phase"]
        p_name = phase["name"]
        pairs = phase["pairs"]
        epochs_per_fold = phase["epochs"]
        assert isinstance(pairs, list)
        
        print(f"\n>>>>> ENTERING PHASE {p_id}: {p_name} <<<<<")
        print(f"      Pairs: {len(pairs)} active")
        
        for fold in range(1, NUM_FOLDS + 1):
            print(f"\n--- Launching Phase {p_id} | Fold {fold}/{NUM_FOLDS} ---")
            
            # Dynamic Epoch Scaling: Monotonically increasing across ALL phases and folds
            global_target_epochs += epochs_per_fold
            
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
