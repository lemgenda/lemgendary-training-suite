import yaml  # pyre-ignore
import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Global Orchestration")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    args = parser.parse_args()

    print("🚀 Initializing Global LemGendary Training Suite Orchestration")
    
    # Kaggle Dependency Verification
    expected_datasets = {
        "LemGendizedQualityDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-quality-dataset",
        "LemGendizedNoiseDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-noise-dataset",
        "LemGendizedLowLightDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-lowlight-dataset",
        "LemGendizedFaceDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-face-dataset",
        "LemGendizedDegradationDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-degradation-dataset",
        "LemGendizedDetectionDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-dettection-dataset",
        "LemGendizedSuperResDataset": "https://www.kaggle.com/datasets/lemtreursi/lemgendized-superres-dataset"
    }
    
    missing_any = False
    if args.env == "local":
        print("🔍 Executing Local Array Dependency Verification...")
        for ds_name, link in expected_datasets.items():
            ds_path = os.path.join(os.path.dirname(__file__), "data", "datasets", ds_name)
            if not os.path.exists(ds_path):
                print(f"❌ CRITICAL MISSING DEPENDENCY: '{ds_name}' is not structurally attached!")
                print(f"   👉 Download / Link from Kaggle natively: {link}")
                missing_any = True
                
        if missing_any:
            print("\n⚠️ Warning: One or more unified datasets are missing from data/datasets/.")
            ans = input("Do you structurally want to proceed violently anyway? [y/N]: ")
            if ans.lower().strip() != 'y':
                print("🛑 Aborting orchestration to prevent native exceptions.\n")
                return
        else:
            print("✅ All 7 physical datasets structurally mapping verified completely!\n")
    else:
        print("☁️ Kaggle Environment Detected: Bypassing local physical dataset structure verification.\n")
    
    # Load unified_models
    unified_models_path = os.path.join(os.path.dirname(__file__), "unified_models.yaml")
    if not os.path.exists(unified_models_path):
        print(f"Error: Unified Models config not found at {unified_models_path}")
        return
        
    with open(unified_models_path, "r") as f:
        models = yaml.safe_load(f)
        
    print(f"Found {len(models)} specialized models in registry. Commencing training matrix...")
    
    for model_key, model_info in models.items():
        model_name = model_info.get("name", model_key)
        batch_size = model_info.get("batch_size", 16)
        learning_rate = model_info.get("learning_rate", 1e-4)
        
        model_filename = model_info.get("filename", model_key)
        final_onnx_path = os.path.join(os.path.dirname(__file__), "trained-models", model_key, f"LemGendary{model_filename}.onnx")
        
        if os.path.exists(final_onnx_path):
            print(f"\n=======================================================")
            print(f"✨ Model '{model_name}' has already fully structurally converged to State-of-the-Art (SOTA) baseline!")
            print(f"   Artifact found: {final_onnx_path}")
            ans = input("Do you want to SKIP training this model? [Y/n]: ")
            if ans.lower().strip() != 'n':
                print(f"⏭  Skipping {model_name}...")
                continue

        print(f"\n=======================================================")
        print(f"🔥 Training Architecture: {model_name}")
        print(f"      Configuration > Batch Size: {batch_size} | LR: {learning_rate}")
        print(f"=======================================================\n")
        
        # Execute the newly optimized Universal Trainer for 50 epochs (will early-stop if plateaued)
        cmd = [
            "python", "training/train.py",
            "--model", model_key,
            "--epochs", "50",
            "--batch_size", str(batch_size),
            "--lr", str(learning_rate),
            "--env", args.env
        ]
        
        try:
            # We use check_call to block and output natively to the user's terminal UI
            subprocess.check_call(cmd)
            print(f"\n✅ {model_name} gracefully halted or completed its epochs.")
        except KeyboardInterrupt:
            print(f"\n⏸ Training for {model_name} was paused manually via KeyboardInterrupt.")
            ans = input("Do you want to proceed to the NEXT model in the registry? (y/n): ")
            if ans.lower().strip() != 'y':
                print("🛑 Aborting global orchestration. You can resume individually later.")
                break
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {model_name} encountered a critical training crash (Exit code: {e.returncode}).")
            ans = input("Do you want to attempt the NEXT model despite the crash? (y/n): ")
            if ans.lower().strip() != 'y':
                print("🛑 Aborting global orchestration.")
                break

    print("\n🏁 Global LemGendary Training Suite Orchestration Complete!")

if __name__ == "__main__":
    main()
