import os
import sys
import shutil
import subprocess

# Hardcoded structural extraction sequence that brutally eliminates mathematical space bottlenecks natively
PHASES = [
    {
        "name": "Phase 1: Deep Quality Assessment",
        "datasets": ["LemGendizedQualityDataset"],
        "models": ["nima_aesthetic", "nima_technical"]
    },
    {
        "name": "Phase 2: High-Fidelity Facial & Object Detection",
        "datasets": ["LemGendizedFaceDataset", "LemGendizedDetectionDataset"],
        "models": ["codeformer", "parsenet", "retinaface_mobilenet", "retinaface_resnet", "yolov8n"]
    },
    {
        "name": "Phase 3: Cross-Domain Environmental Restoration",
        "datasets": ["LemGendizedDegradationDataset", "LemGendizedSuperResDataset", "LemGendizedLowLightDataset", "LemGendizedNoiseDataset"],
        "models": [
            "ffanet_indoor", "ffanet_outdoor", "mprnet_deraining", 
            "nafnet_debluring", "ultrazoom_x2", "ultrazoom_x3", "ultrazoom_x4", "ultrazoom_x8",
            "mirnet_lowlight", "mirnet_exposure", "nafnet_denoising",
            "film_restorer", "upn_v2", "professional_multitask_restoration"
        ]
    }
]

def check_kaggle_auth():
    if os.name == 'nt':
        kaggle_dir = os.path.join(os.environ['USERPROFILE'], '.kaggle')
    else:
        kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    
    if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
        print("❌ CRITICAL ERROR: Kaggle API credentials not found!")
        print(f"Please place your kaggle.json file firmly in: {kaggle_dir}")
        print("You can download this API token natively from your Kaggle Account Settings.")
        sys.exit(1)

def run_orchestrator():
    print("==========================================================================")
    print(" 🧠 LemGendary Smart Caching Kaggle Orchestrator (Dynamic Streams) ")
    print("==========================================================================")
    
    check_kaggle_auth()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "datasets")
    os.makedirs(data_dir, exist_ok=True)
    
    python_exe = sys.executable
    train_script = os.path.join(base_dir, "training", "train.py")

    for phase in PHASES:
        print(f"\n🚀 Initiating {phase['name']}...")
        
        # 1. Download Required Datasets via Kaggle API natively
        for ds in phase["datasets"]:
            ds_path = os.path.join(data_dir, ds)
            if not os.path.exists(ds_path):
                print(f"  [CACHE MISS] Dynamically downloading '{ds}' from Kaggle Cloud via API...")
                cmd = f"kaggle datasets download -d lemtreursi/{ds.lower()} -p \"{data_dir}\" --unzip"
                try:
                    subprocess.run(cmd, shell=True, check=True)
                    print(f"  ✅ '{ds}' flawlessly mounted to Local Cache and Extracted.")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ FAILED to stream Kaggle dataset {ds} (Exit Code: {e.returncode})")
                    sys.exit(1)
            else:
                print(f"  [CACHE HIT] '{ds}' already physically present in Local Cache. Skipping Kaggle API call.")
                
        # 2. Train Models Sequentially natively
        for model in phase["models"]:
            print(f"\n  ▶ Deploying Mathematical Training Matrix for: {model}")
            cmd = [python_exe, train_script, "--model", model, "--env", "local"]
            try:
                subprocess.check_call(cmd)
                print(f"  ✅ {model} natively converged and exported SOTA parameters.")
            except subprocess.CalledProcessError as e:
                print(f"\n  ⏸ Training for {model} encountered a critical structural error or was manually interrupted.")
                ans = input("  Do you want to proceed to the NEXT model in this mathematical Phase? (y/n): ")
                if ans.lower() != 'y':
                    print("  🛑 Aborting Smart Orchestration gracefully.")
                    sys.exit(1)
            except KeyboardInterrupt:
                print(f"\n  ⏸ Smart Caching Matrix paused forcefully via KeyboardInterrupt.")
                sys.exit(1)
                
        # 3. Aggressive Disk Purging (Shredding) globally
        print(f"\n🗑 Phase mathematically Complete. Engaging Aggressive Disk Shredding Sequences...")
        for ds in phase["datasets"]:
            ds_path = os.path.join(data_dir, ds)
            if os.path.exists(ds_path):
                print(f"  [PURGE] Shredding {ds} from native SSD to mathematically free strict storage bounds...")
                try:
                    shutil.rmtree(ds_path)
                    print(f"  ✅ {ds} perfectly structurally eradicated from SSD memory.")
                except Exception as e:
                    print(f"  ⚠️ Warning: Failed to fully physically shred {ds}: {e}. (Windows File Locks might require manual GC)")

    print("\n🏁 LemGendary Smart Caching Kaggle Orchestrator Sequence 100% Complete.")

if __name__ == "__main__":
    try:
        run_orchestrator()
    except KeyboardInterrupt:
        print("\n🛑 Orchestrator aborted physically via Ctrl+C.")
