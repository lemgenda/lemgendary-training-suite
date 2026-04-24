import subprocess
import os
import sys
import argparse
import yaml
import shutil
import time
from typing import TypedDict, List
from data.data_utils import download_and_extract_dataset

class PhaseDef(TypedDict):
    name: str
    datasets: List[str]
    models: List[str]

PHASES: List[PhaseDef] = [
    {"name": "Phase 1: Deep Quality Assessment", "datasets": ["LemGendizedQualityDataset"], "models": ["nima_aesthetic", "nima_technical"]},
    {"name": "Phase 2A: High-Fidelity Facial Analytics", "datasets": ["LemGendizedFaceDataset"], "models": ["codeformer", "parsenet"]},
    {"name": "Phase 2B: Massive Universal Detection", "datasets": ["LemGendizedFaceDataset", "LemGendizedDetectionDataset"], "models": ["retinaface_mobilenet", "retinaface_resnet", "yolov8n"]},
    {"name": "Phase 3A: Super-Resolution Synthesis", "datasets": ["LemGendizedSuperResDataset"], "models": ["ultrazoom_x2", "ultrazoom_x3", "ultrazoom_x4", "ultrazoom_x8"]},
    {"name": "Phase 3B: Degradation Removal Arrays", "datasets": ["LemGendizedDegradationDataset"], "models": ["ffanet_indoor", "ffanet_outdoor", "mprnet_deraining"]},
    {"name": "Phase 3C: Low-Light Recovery", "datasets": ["LemGendizedLowLightDataset"], "models": ["mirnet_lowlight", "mirnet_exposure"]},
    {"name": "Phase 3D: Denoising Networks", "datasets": ["LemGendizedNoiseDataset"], "models": ["nafnet_denoising"]},
    {"name": "Phase 3E: Universal Cross-Domain Restoration", "datasets": ["LemGendizedSuperResDataset", "LemGendizedDegradationDataset", "LemGendizedLowLightDataset", "LemGendizedNoiseDataset"], "models": ["nafnet_debluring", "film_restorer", "upn_v2", "professional_multitask_restoration"]}
]

def check_kaggle_auth():
    if 'KAGGLE_API_TOKEN' in os.environ or 'KAGGLE_USERNAME' in os.environ:
        return
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_token_path = os.path.join(base_dir, ".kaggle_token")
    if os.path.exists(local_token_path):
        with open(local_token_path, "r") as f:
            token = f.read().strip()
            if token:
                os.environ['KAGGLE_API_TOKEN'] = token
                return
    if os.name == 'nt': kaggle_dir = os.path.join(os.environ['USERPROFILE'], '.kaggle')
    else: kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
        print("❌ CRITICAL ERROR: Kaggle API credentials entirely missing!")
        print("Please place kaggle.json in ~/.kaggle/ OR set the KAGGLE_API_TOKEN environment variable natively.")
        sys.exit(1)

def get_future_datasets(current_idx):
    future_datasets = set()
    for i, p in enumerate(PHASES):
        if i > current_idx:
            future_datasets.update(p["datasets"])
    return future_datasets

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    parser = argparse.ArgumentParser(description="Global Orchestration")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--epochs", type=int, default=1000, help="Force override SOTA iterations mathematically natively.")
    parser.add_argument("--yes", action="store_true", help="Automatically bypass interactive prompts for 2026 unit-test orchestration.")
    args = parser.parse_args()

    print("==========================================================================")
    print(" 🚀 Initializing Global LemGendary Training Suite Orchestration")
    print(" 🧠 Smart Caching + Phased Matrix Mode Active")
    print("==========================================================================")
    check_kaggle_auth()
    
    try:
        if not hasattr(yaml, "safe_load"):
            raise AttributeError("Incomplete environment detected.")
    except (NameError, AttributeError):
        print("\n❌ [CRITICAL] Environment Integrity Audit FAILED! Missing PyYAML.")
        return
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    unified_models_path = os.path.join(base_dir, "unified_models.yaml")
    if not os.path.exists(unified_models_path):
        print(f"Error: Unified Models config not found at {unified_models_path}")
        return
        
    with open(unified_models_path, "r") as f:
        registry = yaml.safe_load(f)

    data_dir = os.path.join(base_dir, "data", "datasets")
    os.makedirs(data_dir, exist_ok=True)
    python_exe = sys.executable
    train_script = os.path.join(base_dir, "training", "train.py")

    print("\n--- Intelligent Model Matrix Selection ---")
    active_phases = []
    auto_accept = args.yes
    
    for phase in PHASES:
        approved_models = []
        for model_key in phase["models"]:
            if model_key not in registry: continue
            
            # Check for existing ONNX SOTA artifacts
            model_info = registry.get(model_key, {})
            model_name = model_info.get("name", model_key)
            model_filename = model_info.get("filename", model_key)
            final_onnx_path = os.path.join(base_dir, "trained-models", model_key, f"LemGendary{model_filename}.onnx")
            
            if os.path.exists(final_onnx_path):
                print(f"✨ Model '{model_name}' has already fully structurally converged to SOTA baseline! Skipping...")
                continue

            if auto_accept:
                ans = 'y'
            else:
                ans = input(f"  ▶ Do you want to physically train >> {model_name} << ? (y/n/all): ").strip().lower()
                if ans == 'all':
                    auto_accept = True
                    ans = 'y'
            if ans == 'y':
                approved_models.append(model_key)
                
        if approved_models:
            active_phases.append({
                "name": phase["name"],
                "datasets": phase["datasets"],
                "models": approved_models
            })
            
    if not active_phases:
        print("\n🛑 No mathematical models selected or all models have SOTA artifacts. Terminating Orchestrator natively.")
        sys.exit(0)
        
    print("\n✅ Matrix compiled securely. Booting Cloud Pipeline...")

    for p_idx, phase in enumerate(active_phases):
        print(f"\n🚀 Initiating {phase['name']}...")
        
        for ds in phase["datasets"]:
            ds_path = os.path.join(data_dir, ds)
            lock_file = os.path.join(data_dir, f"{ds}.lock")
            
            if os.path.exists(lock_file):
                print(f"  [CACHE SYNC] Background zero-latency pre-fetch is still finalizing '{ds}'. Waiting for Mutex Lock release legitimately...")
                while os.path.exists(lock_file): time.sleep(5)
                print(f"  ✅ '{ds}' stream mathematically resolved in the background!")
                
            if not os.path.exists(ds_path) and args.env == "local":
                if not download_and_extract_dataset(ds, data_dir):
                    print(f"  ❌ FAILED to stream Kaggle dataset {ds} natively.")
                    if args.yes:
                        ans = 'y'
                    else:
                        ans = input("Do you structurally want to proceed violently anyway? [y/N]: ")
                    if ans.lower().strip() != 'y':
                        sys.exit(1)
            else:
                print(f"  [CACHE HIT] '{ds}' already locally resident in SSD Cache.")

        for m_idx, model_key in enumerate(phase["models"]):
            model_info = registry.get(model_key, {})
            model_name = model_info.get("name", model_key)
            print(f"\n=======================================================")
            print(f"🔥 Training Architecture: {model_name} ({model_key})")
            print(f"=======================================================\n")
            
            cmd = [python_exe, train_script, "--model", model_key, "--epochs", str(args.epochs), "--env", args.env]
            
            # --- NEXT PHASE BACKGROUND PRE-FETCH LOOKAHEAD ---
            if m_idx == len(phase["models"]) - 1 and p_idx < len(active_phases) - 1:
                next_phase = active_phases[p_idx + 1]
                prefetch_list = []
                for next_ds in next_phase["datasets"]:
                    next_ds_path = os.path.join(data_dir, next_ds)
                    if not os.path.exists(next_ds_path):
                        prefetch_list.append(f"lemtreursi/{next_ds.lower()}:{next_ds}")
                if prefetch_list:
                    cmd.extend(["--prefetch_datasets", ",".join(prefetch_list)])
                    print(f"  [Orchestrator] Armed {model_name} with Zero-Latency Pre-Fetch Trigger perfectly optimized.")

            try:
                subprocess.check_call(cmd)
                print(f"\n✅ {model_name} natively converged and exported SOTA parameters.")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ {model_name} encountered a critical structural error (Exit code: {e.returncode}).")
                ans = input("  Do you want to attempt the NEXT model despite the crash? (y/n): ")
                if ans.lower().strip() != 'y': sys.exit(1)
            except KeyboardInterrupt:
                print(f"\n⏸ Smart Caching Matrix paused forcefully via KeyboardInterrupt.")
                ans = input("  Do you want to proceed to the NEXT model in this mathematical Phase? (y/n): ")
                if ans.lower().strip() != 'y': sys.exit(1)

        # 3. Aggressive Disk Purging globally specifically based on remaining timeline
        future_reqs = get_future_datasets(p_idx)
        print(f"\n🗑 Phase mathematically Complete. Executing Surgical Memory Purger...")
        for ds in phase["datasets"]:
            if ds not in future_reqs:
                ds_path = os.path.join(data_dir, ds)
                if os.path.exists(ds_path):
                    print(f"  [PURGE] Shredding dead dataset {ds} from native SSD to mathematically free space...")
                    try:
                        shutil.rmtree(ds_path)
                    except Exception as e:
                        pass
            else:
                print(f"  [SAVE] Keeping {ds} heavily cached natively. Required by future Orchestration Sequences!")

    print("\n🏁 Global LemGendary Training Suite Orchestration Complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Orchestrator aborted physically via Ctrl+C.")
