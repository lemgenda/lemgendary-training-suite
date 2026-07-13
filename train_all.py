import subprocess
import os
import sys

# --- 2026: kagglesdk Dependency Hardening (ImportError Patch) ---
try:
    import kagglesdk.kaggle_env as ke
    if not hasattr(ke, 'get_web_endpoint'):
        def get_web_endpoint(env):
            endpoint = ke.get_endpoint(env) if hasattr(ke, 'get_endpoint') else "https://api.kaggle.com"
            if "api.kaggle.com" in endpoint:
                return "https://www.kaggle.com"
            return endpoint
        ke.get_web_endpoint = get_web_endpoint
except ImportError:
    pass
import argparse
import yaml
import shutil
import time
import json
from typing import TypedDict, List
from data.data_utils import download_and_extract_dataset
from datetime import datetime

# [SENIOR HARDENING v16.0 - SYNC_ID: 1512]

class PhaseDef(TypedDict):
    name: str
    datasets: List[str]
    models: List[str]

PHASES: List[PhaseDef] = [
    {"name": "Phase 1: Deep Quality & Safety Assessment", "datasets": ["LemGendizedQualityDataset", "ClassificationMasterManifold"], "models": ["nima_aesthetic", "nima_technical", "nima_authenticity", "anime_nsfw_classification"]},
    {"name": "Phase 2A: High-Fidelity Facial Analytics", "datasets": ["LemGendizedFaceDataset"], "models": ["codeformer", "parsenet"]},
    {"name": "Phase 2B: Massive Universal Detection", "datasets": ["LemGendizedFaceDataset", "LemGendizedDetectionDataset"], "models": ["retinaface_mobilenet", "retinaface_resnet", "yolov8n"]},
    {"name": "Phase 3A: Master Super-Resolution Synthesis", "datasets": ["LemGendizedSuperResDataset"], "models": ["ultrazoom"]},
    {"name": "Phase 3B: Degradation Removal Arrays", "datasets": ["LemGendizedDegradationDataset"], "models": ["ffanet_indoor", "ffanet_outdoor", "mprnet_deraining"]},
    {"name": "Phase 3C: Low-Light Recovery", "datasets": ["LemGendizedLowLightDataset"], "models": ["mirnet_lowlight", "mirnet_exposure"]},
    {"name": "Phase 3D: Denoising Networks", "datasets": ["LemGendizedNoiseDataset"], "models": ["nafnet_denoising"]},
    {"name": "Phase 3E: Universal Cross-Domain Restoration", "datasets": ["LemGendizedSuperResDataset", "LemGendizedDegradationDataset", "LemGendizedLowLightDataset", "LemGendizedNoiseDataset"], "models": ["nafnet_debluring", "film_restorer", "upn_v2", "professional_multitask_restoration"]},
    {"name": "Phase 4: Master Generative Manifolds", "datasets": ["diffusion_master_manifold"], "models": ["diffusion_sdxl", "diffusion_flux"]},
    {"name": "Phase 5: Master Multimodal Reasoning", "datasets": ["vision_language_master_manifold"], "models": ["vlm_llava", "vlm_blip2"]}
]

def main():
    parser = argparse.ArgumentParser(description="Global LemGendary Fleet Orchestrator (v16.0 Nuclear)")
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="Task 11.1: Bypass SOTA existence checks.")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*80)
    print(" 🚀 NUCLEAR FLEET ORCHESTRATOR v16.0")
    print(" 🧠 Sequential Manifold Execution + Driver Cooldown Enabled")
    print("="*80 + "\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    unified_models_path = os.path.join(base_dir, config.get("unified_models", "unified_models_v2.yaml"))
    with open(unified_models_path, "r") as f:
        registry = yaml.safe_load(f)

    data_dir = config.get("datasets_dir", "data/datasets")
    train_script = os.path.join(base_dir, "training", "train.py")
    
    # --- Task 11.2: Failure Matrix Initialization ---
    failure_log_path = os.path.join(base_dir, "fleet_failure_report.json")
    failure_report = {"timestamp": datetime.now().isoformat(), "failures": []}

    active_phases = []
    auto_accept = args.yes

    # --- Task 11.4: Dynamic Model Extension ---
    # Automatically schedule any models in the registry that were missed by the hardcoded PHASES array
    scheduled_models = set(m for p in PHASES for m in p["models"])
    unscheduled = [m for m, info in registry.items() if isinstance(info, dict) and m not in scheduled_models and not m.startswith('_')]
    if unscheduled:
        PHASES.append({"name": "Phase 6: Dynamic Fleet Extensions", "datasets": [], "models": unscheduled})
    
    for phase in PHASES:
        approved_models = []
        for model_key in phase["models"]:
            if model_key not in registry: continue
            
            # SOTA Skip Logic (Task 11.1)
            model_info = registry[model_key]
            model_filename = model_info.get("filename", model_key)
            final_pth = os.path.join(base_dir, "trained-models", model_key, f"{model_key}_best.pth")
            
            if os.path.exists(final_pth) and not args.force:
                print(f"✨ [SKIP] Model '{model_key}' has existing SOTA artifacts. Use --force to re-train.")
                continue

            # Diagnostic Skip Logic (Option 4)
            if args.epochs == 1:
                export_dir = os.path.abspath(os.path.join(base_dir, "..", "LemGendaryModels", model_key))
                if os.path.exists(export_dir):
                    files = os.listdir(export_dir)
                    has_onnx = any(f.endswith('.onnx') for f in files)
                    has_pt = any(f.endswith('.pt') or f.endswith('.pth') for f in files)
                    if has_onnx and has_pt:
                        print(f"✨ [SKIP] Model '{model_key}' already has exported ONNX and PT binaries (Single-Epoch Test Bypass).")
                        continue

            if auto_accept or input(f"▶ Train >> {model_key} << ? (y/n/all): ").strip().lower() in ['y', 'all']:
                approved_models.append(model_key)
                if not auto_accept and 'all' in sys.stdin.readline(): auto_accept = True
                
        if approved_models:
            active_phases.append({"name": phase["name"], "datasets": phase["datasets"], "models": approved_models})

    for p_idx, phase in enumerate(active_phases):
        print(f"\n⚡ Initiating {phase['name']}...")
        
        for m_idx, model_key in enumerate(phase["models"]):
            print(f"\n" + "-"*60)
            print(f"🔥 MATRIX: {model_key}")
            print("-"*60 + "\n")
            
            cmd = [sys.executable, train_script, "--model", model_key, "--epochs", str(args.epochs), "--env", args.env]
            
            try:
                subprocess.check_call(cmd)
                print(f"\n✅ {model_key} converged.")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ {model_key} structural failure. Logging to report.")
                failure_report["failures"].append({"model": model_key, "phase": phase["name"], "code": e.returncode})
                with open(failure_log_path, 'w') as f: json.dump(failure_report, f, indent=4)
                if not args.yes and input("Proceed to next? (y/n): ").lower() != 'y': sys.exit(1)
            
            # --- Task 11.3: Driver Cooldown ---
            print("💤 [COOLDOWN] Reclaiming physical VRAM manifold...")
            time.sleep(2)

    # --- Task 11.4: Global Dashboard Hook ---
    print("\n🏁 Nuclear Orchestration Complete!")
    if failure_report["failures"]:
        print(f"⚠️ Warning: {len(failure_report['failures'])} models failed. See {failure_log_path}")

if __name__ == "__main__":
    main()
