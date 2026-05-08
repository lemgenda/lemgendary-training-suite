import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="LemGendary Checkpoint Sync v1.0")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--target", type=str, default="/kaggle/working/export")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 2026: Auto-resolve model name for persistence structure
    model_name_formatted = f"Lemgendary_{args.model.replace('_', ' ').title().replace(' ', '_')}_Checkpoints"
    persistence_root = os.path.join(args.target, model_name_formatted)
    os.makedirs(persistence_root, exist_ok=True)
    
    # 1. Sync metrics.csv
    src_metrics = os.path.join(project_root, "metrics.csv")
    if os.path.exists(src_metrics):
        shutil.copy2(src_metrics, os.path.join(persistence_root, "metrics.csv"))
        print(f"✅ Synced metrics.csv -> {persistence_root}")

    # 2. Sync Checkpoints
    src_ckpt_dir = os.path.join(project_root, "checkpoints")
    dst_ckpt_dir = os.path.join(persistence_root, "checkpoints")
    os.makedirs(dst_ckpt_dir, exist_ok=True)
    
    if os.path.exists(src_ckpt_dir):
        for f in os.listdir(src_ckpt_dir):
            if f.endswith('.pth') and args.model in f:
                shutil.copy2(os.path.join(src_ckpt_dir, f), os.path.join(dst_ckpt_dir, f))
                print(f"✅ Synced {f} -> {dst_ckpt_dir}")

if __name__ == "__main__":
    main()
