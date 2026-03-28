import os
import sys
import shutil
import zipfile
import psutil
from tqdm import tqdm
import kaggle # pyre-ignore
from kaggle.api.kaggle_api_extended import KaggleApi # pyre-ignore
from typing import TypedDict, List

class PhaseDef(TypedDict):
    name: str
    datasets: List[str]
    models: List[str]

# Hardcoded hyper-optimized execution sequence structured mathematically to explicitly minimize Peak Disk Constraints natively.
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

def check_disk_space(required_bytes, target_path):
    """
    Analyzes the structural occupancy of the target physical volume to ensure 
    safe extraction of massive isolated Kaggle datasets natively.
    """
    free_bytes = psutil.disk_usage(target_path).free
    # We require 2.5x the compressed size to hold the .zip + extracted array + buffer
    needed_with_buffer = int(required_bytes * 2.5)
    
    if free_bytes < needed_with_buffer:
        free_gb = free_bytes / (1024**3)
        needed_gb = needed_with_buffer / (1024**3)
        print(f"\n⚠️  [DISK SPACE WARNING] Low volume detected on target topological array!")
        print(f"   Available: {free_gb:.2f} GB | Minimum Needed (with buffer): {needed_gb:.2f} GB")
        ans = input("   👉 Do you want to proceed violently anyway? (y/n): ").strip().lower()
        if ans != 'y':
            print("🛑 Operation aborted by user due to Disk Constraint.")
            sys.exit(1)
    return True

def check_kaggle_auth():
    if 'KAGGLE_API_TOKEN' in os.environ or 'KAGGLE_USERNAME' in os.environ:
        return # Natively authenticated via modern system environment variables
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_token_path = os.path.join(base_dir, ".kaggle_token")
    if os.path.exists(local_token_path):
        with open(local_token_path, "r") as f:
            token = f.read().strip()
            if token:
                os.environ['KAGGLE_API_TOKEN'] = token
                return # Structurally injected from local repository cache
                
    if os.name == 'nt': kaggle_dir = os.path.join(os.environ['USERPROFILE'], '.kaggle')
    else: kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
        print("❌ CRITICAL ERROR: Kaggle API credentials entirely missing!")
        print("Please place kaggle.json in ~/.kaggle/ OR heavily set the KAGGLE_API_TOKEN environment variable natively.")
        sys.exit(1)

def get_future_datasets(current_idx):
    future_datasets = set()
    for i, p in enumerate(PHASES):
        if i > current_idx:
            future_datasets.update(p["datasets"])
    return future_datasets

def run_orchestrator():
    print("==========================================================================")
    print(" 🧠 LemGendary Smart Caching Kaggle Orchestrator (Dynamic Streams) ")
    print("==========================================================================")
    check_kaggle_auth()
    
    print("\n--- Intelligent Model Matrix Selection ---")
    print("Type 'y' to train, 'n' to skip, or 'all' to blindly auto-accept the rest natively.")
    active_phases: List[PhaseDef] = []
    auto_accept = False
    
    for phase in PHASES:
        approved_models = []
        for m in phase["models"]:
            if auto_accept:
                ans = 'y'
            else:
                ans = input(f"  ▶ Do you want to physically train >> {m} << ? (y/n/all): ").strip().lower()
                if ans == 'all':
                    auto_accept = True
                    ans = 'y'
            if ans == 'y':
                approved_models.append(m)
        if approved_models:
            active_phases.append({
                "name": phase["name"],
                "datasets": phase["datasets"],
                "models": approved_models
            })
    
    if not active_phases:
        print("\n🛑 No mathematical models selected. Terminating Orchestrator natively.")
        sys.exit(0)
        
    print("\n✅ Matrix compiled securely. Booting Cloud Pipeline...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "datasets")
    os.makedirs(data_dir, exist_ok=True)
    
    python_exe = sys.executable
    train_script = os.path.join(base_dir, "training", "train.py")
    import time
    
    for p_idx, phase in enumerate(active_phases):
        print(f"\n🚀 Initiating {phase['name']}...")
        
        for ds in phase["datasets"]:
            ds_path = os.path.join(data_dir, ds)
            lock_file = os.path.join(data_dir, f"{ds}.lock")
            
            if os.path.exists(lock_file):
                print(f"  [CACHE SYNC] Background zero-latency pre-fetch is still finalizing '{ds}'. Waiting for Mutex Lock release legitimately...")
                while os.path.exists(lock_file): time.sleep(5)
                print(f"  ✅ '{ds}' stream mathematically resolved in the background!")
                
            if not os.path.exists(ds_path):
                # 2026 Native API Resolution
                api = KaggleApi()
                api.authenticate()
                
                # Metadata check for Disk Verification
                ds_slug = f"lemtreursi/{ds.lower()}"
                try:
                    meta = api.dataset_metadata(ds_slug)
                    total_bytes = meta.get('size', 1024**3) # Fallback to 1GB
                    check_disk_space(total_bytes, data_dir)
                except Exception:
                    print(f"  [!] Metadata check failed for {ds}. Using optimistic disk mapping...")
                    total_bytes = 1024**3
                
                # ⬇️ Bit-Level Download with Progress Bar
                zip_filename = f"{ds.lower()}.zip"
                zip_path = os.path.join(data_dir, zip_filename)
                
                print(f"  [CACHE MISS] Executing Downlink: {ds}...")
                pbar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc=f"    ⏬ {ds}")
                
                def progress_callback(bytes_delta):
                    pbar.update(bytes_delta)

                try:
                    api.dataset_download_files(ds_slug, path=data_dir, quiet=True)
                    # Note: kaggle-api-extended doesn't easily expose chunked callbacks for datasets.
                    # We'll pulse the pbar or use a more advanced chunked method if available.
                    # As a SOTA 2026 fallback, we'll confirm the zip size.
                    pbar.close()
                    
                    # 📦 File-Level Extraction with Progress Bar
                    if os.path.exists(zip_path):
                        print(f"  [UNPACK] Synchronizing topological array: {ds}...")
                        with zipfile.ZipFile(zip_path, 'r') as z:
                            files = z.namelist()
                            with tqdm(total=len(files), unit='file', desc=f"    📦 Unpacking") as upbar:
                                for f in files:
                                    z.extract(f, ds_path)
                                    upbar.update(1)
                        os.remove(zip_path) # Clean up zip artifact
                except Exception as e:
                    print(f"  ❌ FAILED to stream Kaggle dataset {ds}: {str(e)}")
                    sys.exit(1)
            else:
                print(f"  [CACHE HIT] '{ds}' already locally resident in SSD Cache.")
                
        # 2. Train Models Sequentially natively
        for m_idx, model in enumerate(phase["models"]):
            print(f"\n  ▶ Deploying Mathematical Training Matrix for: {model}")
            cmd = [python_exe, train_script, "--model", model, "--env", "local"]
            
            # --- NEXT PHASE BACKGROUND PRE-FETCH LOOKAHEAD ---
            if m_idx == len(phase["models"]) - 1 and p_idx < len(active_phases) - 1:
                next_phase = active_phases[p_idx + 1]  # pyre-ignore
                prefetch_list = []
                for next_ds in next_phase["datasets"]:
                    next_ds_path = os.path.join(data_dir, next_ds)
                    # Tell train.py to gracefully spawn a download for any datasets missing from upcoming phase
                    if not os.path.exists(next_ds_path):
                        prefetch_list.append(f"lemtreursi/{next_ds.lower()}:{next_ds}")
                if prefetch_list:
                    cmd.extend(["--prefetch_datasets", ",".join(prefetch_list)])
                    print(f"  [Orchestrator] Armed {model} with Zero-Latency Pre-Fetch Trigger perfectly optimized.")

            try:
                subprocess.check_call(cmd)
                print(f"  ✅ {model} natively converged and exported SOTA parameters.")
            except subprocess.CalledProcessError as e:
                print(f"\n  ⏸ Training for {model} encountered a critical structural error or was manually interrupted.")
                ans = input("  Do you want to proceed to the NEXT model in this mathematical Phase? (y/n): ")
                if ans.lower() != 'y': sys.exit(1)
            except KeyboardInterrupt:
                print(f"\n  ⏸ Smart Caching Matrix paused forcefully via KeyboardInterrupt.")
                sys.exit(1)
                
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

    print("\n🏁 LemGendary Smart Caching Kaggle Orchestrator Sequence 100% Complete.")

if __name__ == "__main__":
    try:
        run_orchestrator()
    except KeyboardInterrupt:
        print("\n🛑 Orchestrator aborted physically via Ctrl+C.")
