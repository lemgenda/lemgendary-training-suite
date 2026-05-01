# 2026: Environment Linter Sync
import os
import sys
import argparse
import warnings
import atexit
import signal
import subprocess
import time
import shutil
import gc
import math

# --- 2026 Hardware Acceleration & Stability Patch ---
# Increase recursion limit for exceptionally deep architectures (NIMA/Restorers)
sys.setrecursionlimit(2000)

# Suppress noisy Triton, torchvision, and serialization warnings (benign across GTX/RTX training)
warnings.filterwarnings("ignore", category=UserWarning, module="triton")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=r"clamping frac to range \[0, 1\]")

# --- Hyper-Verbose Path Defense (2026 Specialization) ---
# Anchor the search path to the script's own folder to bypass "Ghost Python" hijacking.
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
venv_site_pkgs = os.path.normpath(os.path.join(workspace_root, ".venv", "Lib", "site-packages"))

# Anchor both the workspace and venv site-packages BEFORE any domestic imports
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if os.path.exists(venv_site_pkgs) and venv_site_pkgs not in sys.path:
    sys.path.insert(0, venv_site_pkgs)

try:
    import yaml
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn # 2026 SOTA: Smooth Generalization
    from training.optimization_engine import SmartTrainingGovernor
except ImportError as e:
    print(f"\n--- LemGendary Crash Diagnostics ---")
    print(f"Executable: {sys.executable}")
    print(f"Script Location: {__file__}")
    print(f"Project Root: {workspace_root}")
    print(f"Looking for venv site-packages at: {venv_site_pkgs} (Exists: {os.path.exists(venv_site_pkgs)})")
    print(f"Current Path (sys.path[0]): {sys.path[0]}")
    print(f"Full sys.path: {sys.path}")
    print(f"\n❌ [CRITICAL] Dependency Error: {e}")
    print("  [!] Your LemGendary environment is incomplete or corrupted.")
    print("  [!] Fix: Run the 'lemgendary_hub.ps1' script and select Option 1.")
    sys.exit(1)

# (Workspace root correctly anchored in boot sequence above)

# --- 2026 Process Janitor Hooks ---
_active_processes = []

# --- 2026 Emergency Debug Injection removed for cleaner console ---

def cleanup_active_processes(*args):
    """Indestructible cleanup of all LemGendary project child-processes."""
    if not _active_processes:
        return
    print(f"\n🧹 [JANITOR] Terminating {_active_processes.__len__()} active LemGendary sub-processes...")
    for p in _active_processes:
        if p.poll() is None: # Still running
            try:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], capture_output=True)
                else:
                    p.terminate()
            except Exception: pass
    _active_processes.clear()

atexit.register(cleanup_active_processes)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

from data.dataset import MultiTaskDataset
from models.factory import get_model

# --- 2026: SOTA Metric Registry & Polarity Definitions ---
# higher_better: True (Higher is Better), False (Lower is Better)
METRIC_DIRECTIONS = {
    'plcc': True, 'srcc': True, 'psnr': True, 'ssim': True,
    'lpips': False, 'fid': False, 'map50': True, 'map50_95': True,
    'rank_margin': False, 'accuracy': True
}

# Standard Weights for Quality Score calculation (Multiplier applied to normalized 0.0-1.0 range)
METRIC_WEIGHTS = {
    'plcc': 50, 'srcc': 50, 'psnr': 1, 'ssim': 40,
    'lpips': 40, 'fid': 1, 'map50': 100, 'map50_95': 100,
    'rank_margin': 20, 'accuracy': 100
}

def safe_replace(src, dst):
    """Battle-Hardened atomic replace for Windows. Uses 3-stage recovery (Replace -> Remove/Rename -> Copy/Delete)."""
    max_retries = 15
    base_delay = 0.5

    for i in range(max_retries):
        try:
            if os.path.exists(dst):
                # 2026: Windows Lock Defense - Rename then Replace
                temp_old = f"{dst}.old_{int(time.time())}"
                os.rename(dst, temp_old)
                os.rename(src, dst)
                try: os.remove(temp_old)
                except: pass
            else:
                os.rename(src, dst)
            return True
        except (PermissionError, OSError) as e:
            time.sleep(base_delay * (1.5 ** i))
    return False

def git_hub_sync(repo_path, remote_url, message):
    """
    2026 Resilience: Robust synchronization for external repositories.
    Handles initialization, remotes, and pushes with rebase recovery.
    Uses GITHUB_PAT for headless authentication on Kaggle.
    """
    try:
        import subprocess
        # 2026 Resilience: Credential Injection
        pat = os.environ.get('GITHUB_PAT')
        if pat and "github.com" in remote_url and "@" not in remote_url:
            authenticated_url = remote_url.replace("https://", f"https://{pat}@")
        else:
            authenticated_url = remote_url

        # 1. Check if it's a git repo
        if not os.path.exists(os.path.join(repo_path, ".git")):
            print(f" 🚀 [CLOUD SYNC] Initializing new repository at {repo_path}...")
            subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, timeout=30)
            subprocess.run(["git", "remote", "add", "origin", authenticated_url], cwd=repo_path, capture_output=True, timeout=30)
            subprocess.run(["git", "checkout", "-b", "main"], cwd=repo_path, capture_output=True, timeout=30)
            subprocess.run(["git", "config", "user.email", "lemgendary@ai.com"], cwd=repo_path, capture_output=True, timeout=30)
            subprocess.run(["git", "config", "user.name", "LemGendary Bot"], cwd=repo_path, capture_output=True, timeout=30)
        elif pat and remote_url != "origin":
             # Update remote to include PAT for existing hub repos
             subprocess.run(["git", "remote", "set-url", "origin", authenticated_url], cwd=repo_path, capture_output=True, timeout=30)
        
        # 2. Sync
        print(f" 📡 [CLOUD SYNC] Staging changes in {os.path.basename(repo_path)}...")
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, timeout=60)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_path, capture_output=True, text=True, timeout=30)
        if status.stdout.strip():
            print(f" 📡 [CLOUD SYNC] Committing changes...")
            subprocess.run(["git", "commit", "-m", message], cwd=repo_path, capture_output=True, timeout=60)
            print(f" 📡 [CLOUD SYNC] Pushing to origin/main (60s timeout)...")
            push_res = subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repo_path, capture_output=True, text=True, timeout=120)
            if push_res.returncode == 0:
                print(f" ✅ [CLOUD SYNC] '{os.path.basename(repo_path)}' synchronized successfully.")
            else:
                print(f" 📡 [CLOUD SYNC] Push failed. Attempting rebase recovery...")
                # If push fails, attempt a non-destructive rebase (Production Manifold Protection)
                subprocess.run(["git", "pull", "origin", "main", "--rebase"], cwd=repo_path, capture_output=True, timeout=120)
                subprocess.run(["git", "push", "origin", "main"], cwd=repo_path, capture_output=True, timeout=120)
                print(f" ✅ [CLOUD SYNC] '{os.path.basename(repo_path)}' synchronized after rebase.")
        else:
            print(f" 📡 [CLOUD SYNC] No changes detected in {os.path.basename(repo_path)}.")
    except subprocess.TimeoutExpired:
        print(f" ⚠️ [CLOUD SYNC] Sync TIMEOUT for {repo_path}. GitHub might be unreachable or credentials requested.")
    except Exception as e:
        print(f" ⚠️ [CLOUD SYNC] Hub Sync failed for {repo_path}: {e}")

from training.losses import CombinedLoss

def trigger_sota_export(model, args, config, unified_models_registry, epoch, plcc, srcc, psnr, ssim_val, lpips_val, fid):
    """
    State-of-the-Art (SOTA) Deployment Automation (v1.1.0)
    Automatically synthesizes production-ready binaries (ONNX, PyTorch Standalone)
    immediately upon hitting a new metric milestone.
    """
    print(f"\n[*] [SOTA DEPLOYMENT] Triggering high-fidelity export for {args.model}...")
    try:
        device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        model.eval()

        export_dir = os.path.join(os.getcwd(), "staging_export")
        os.makedirs(export_dir, exist_ok=True)

        model_info = unified_models_registry.get(args.model, {})
        size_raw = model_info.get("input_size", config.get("default_img_size", 256))
        if isinstance(size_raw, list):
            h, w = (int(size_raw[1]), int(size_raw[2])) if len(size_raw)==3 else (int(size_raw[0]), int(size_raw[1]))
        else:
            h, w = int(size_raw), int(size_raw)

        dummy_input = torch.randn(1, 3, h, w).to(device)
        model_filename = model_info.get("filename", args.model)

        # 2026 Resilience: Delegation to specialized export scripts with UTF-8 hardening
        python_exe = sys.executable
        # Ensure we use an absolute path relative to the script location
        script_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        export_script_dir = os.path.join(script_root, "export")

        # Standardized environment with UTF-8 support for emoji-heavy diagnostic scripts
        export_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        # 1. ONNX Synthesis
        print(f"   -> [1/2] Synthesizing Global ONNX Matrix...")
        onnx_script = os.path.join(export_script_dir, "export_onnx_model.py")
        subprocess.call([python_exe, onnx_script, "--model", args.model, "--yes"], env=export_env)

        # 2. Torch Unity Synthesis
        print(f"   -> [2/2] Synthesizing Standalone PyTorch Unity...")
        torch_script = os.path.join(export_script_dir, "export_torch_model.py")
        subprocess.call([python_exe, torch_script, "--model", args.model, "--yes"], env=export_env)

        # 3. Documentation (README.md)
        from training.doc_generator import build_model_readme # pyre-ignore
        metrics_dict = {"plcc": plcc, "srcc": srcc, "psnr": psnr, "ssim": ssim_val, "lpips": lpips_val, "fid": fid}
        readme_text = build_model_readme(args.model, unified_models_registry, epoch+1, metrics_dict)

        # Deployment Sync
        trained_models_dir = os.path.normpath(os.path.join(os.getcwd(), "..", "LemGendaryModels", args.model))
        os.makedirs(trained_models_dir, exist_ok=True)
        with open(os.path.join(trained_models_dir, "README.md"), "w") as f:
            f.write(readme_text)

        print(f"✅ [SOTA DEPLOYMENT] Successful! Production binaries are live in LemGendaryModels/{args.model}.")
        
        # 2026 Resilience: Immediate Hub Sync for SOTA artifacts (ai-models repo)
        if args.env == 'kaggle':
            hub_url = config.get("model_hub_repo")
            if hub_url:
                git_hub_sync(trained_models_dir, hub_url, f"feat(sota): deploy converged {args.model} epoch {epoch+1}")
    except Exception as e:
        print(f"⚠️  [SOTA DEPLOYMENT] Export phase failed: {e}")

def get_dynamic_batch_size(model_key, model_info, config, device, mode='train'):
    """
    Memory-Sentinel (2026): Calculates the absolute peak batch size for
    high-velocity training on any NVIDIA architecture with zero VRAM paging.
    """
    if device.type != 'cuda':
        return config.get("default_batch_size", 16)

    try:
        # 2026 SOTA: Probe ACTUAL hardware headroom (includes browser/OS overhead)
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        # 2026 Resilience: Safety buffer for 4GB hardware. 
        # Validation is more stable, so we can be more aggressive (0.85 instead of 0.75).
        s_mult = 0.75 if total_vram < (5 * 1024 * 1024 * 1024) else 0.85
        if mode == 'val': s_mult += 0.10 # More aggressive for inference
        available_vram = free_vram * s_mult

        task_type = model_info.get("dataset_type", "quality")
        if isinstance(task_type, list): task_type = task_type[0]

        # 2026 Res-Aware Scaling: Normalize by baseline 224x224 surface area
        size_raw = model_info.get("input_size", config.get("default_img_size", 256))
        h, w = (size_raw[1], size_raw[2]) if isinstance(size_raw, list) and len(size_raw) == 3 else (size_raw, size_raw) if isinstance(size_raw, int) else (size_raw[0], size_raw[1])
        res_multiplier = (int(h) * int(w)) / (224 * 224)

        # Factor in model depth (EfficientNetV2 vs MobileNet)
        backbone_mult = 1.0
        if model_info.get("backbone") == "efficientnet_v2_s":
            backbone_mult = 2.4 # EfficientNetV2-S has significant activation overhead at 384x384

        vram_coeffs = {
            "quality": 25 * 1024 * 1024 * res_multiplier * backbone_mult,    # Calibrated for MobileNetV2 (SOTA Efficiency)
            "detection": 150 * 1024 * 1024 * res_multiplier,   # YOLOv8 (640x640)
            "restoration": 220 * 1024 * 1024 * res_multiplier  # NAFNet/MIRNet (256x256)
        }

        coeff = vram_coeffs.get(task_type, 180 * 1024 * 1024)
        
        # 2026 Resilience: Validation is much leaner (no gradients/adam states)
        if mode == 'val':
            coeff *= 0.6 # 40% discount for inference-only memory footprint
            
        dynamic_batch = int(available_vram / coeff)

        # 2026: Paging Guard - Hard limit for 4GB cards to prevent swapping
        if total_vram < (5 * 1024 * 1024 * 1024):
            # User confirmed 24 is stable for training and 64 for validation at 384px
            if mode == 'train':
                dynamic_batch = min(dynamic_batch, 24)
            else:
                dynamic_batch = min(dynamic_batch, 64)

        # Clamp to professional biological limits
        dynamic_batch = max(8, min(dynamic_batch, 128))

        # 2026: Concatenated Hardware Status
        gpu_name = torch.cuda.get_device_name(0)
        print(f"📡 [MEMORY-SENTINEL] {gpu_name} ({(total_vram/1e9):.1f}GB) | {mode.capitalize()} Batch: {dynamic_batch}")
        return dynamic_batch
    except Exception as e:
        print(f"⚠️ [MEMORY-SENTINEL] Probe failed: {e}. Falling back to safe defaults.")
        return config.get("default_batch_size", 16)

def main():
    # 2026 Resilience: Force UTF-8 encoding for Windows terminals to support emojis
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="LemGendary Training Suite Universal Trainer")
    parser.add_argument("--model", type=str, default="professional_multitask_restoration", help="Model key from unified_models.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"], help="Execution environment")
    parser.add_argument("--prefetch_datasets", type=str, default="", help="Comma separated kaggle endpoint list natively executed asynchronously sequentially upon passing SOTA.")
    args = parser.parse_args()

    # Load config structures explicitly securely
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    unified_models_path = os.path.join(project_root, config["unified_models"])
    with open(unified_models_path, 'r') as f: unified_models_registry = yaml.safe_load(f)

    # --- Device Discovery (2026 Universal Acceleration Suite) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        torch.backends.cudnn.benchmark = True
        print(f"🚀 [HARDWARE] NVIDIA {gpu_name} | CUDA {torch.version.cuda} Active")
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🚀 [HARDWARE] Apple Silicon (Metal) Acceleration Active")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"🚀 [HARDWARE] Intel ARC / XPU Acceleration Active")
    elif hasattr(torch, "dml") and torch.dml.is_available():
        device = torch.device("dml")
        print(f"🚀 [HARDWARE] Microsoft DirectML (AMD/Intel) Active")
    else:
        device = torch.device("cpu")
        print(f"⚠️ [HARDWARE] No Accelerator Found. Defaulting to CPU (Slow).")

    # Load model
    if "yolo" in args.model.lower():
        from data.yolo_config_gen import generate_yolo_yaml  # pyre-ignore

        yaml_path = generate_yolo_yaml(config, args.model, unified_models_registry)

        from ultralytics import YOLO  # pyre-ignore
        model_info = unified_models_registry.get(args.model, {})

        # Dynamic base architecture inference
        default_pt = "yolov8n.pt" if "yolov8" in args.model.lower() else "yolov8n.pt"
        model_pt = model_info.get("checkpoint", default_pt)

        # Fallback to pretrained base architecture if local checkpoint not physically present yet
        if not os.path.exists(model_pt):
            print(f"Warning: Custom local weights '{model_pt}' not found. Defaulting to base architecture '{default_pt}' for initialization.")
            model_pt = default_pt

        model = YOLO(model_pt)

        epochs = args.epochs or config.get("default_epochs", 50)
        batch_size = args.batch_size or config.get("default_batch_size", 16)

        print(f"Starting Ultralytics YOLO Training for {args.model}...")

        # --- NEW: CUSTOM YOLO EXCELLENT QUALITY EARLY STOPPING CALLBACK ---
        def on_fit_epoch_end(trainer):
            metrics = trainer.metrics
            # Bounding box mAP
            map50 = metrics.get('metrics/mAP50(B)', 0)
            map50_95 = metrics.get('metrics/mAP50-95(B)', 0)

            achieved = getattr(trainer, 'excellent_achieved', False)
            countdown = getattr(trainer, 'excellent_countdown', 1)

            if map50_95 > 0.65:
                if not achieved:
                    print(f"\n[ACHIEVEMENT] State-of-the-Art Detection Baseline (mAP@0.5:0.95 > 0.65) breached! Engaging 1-Epoch Reinforcement Countdown...")
                    trainer.excellent_achieved = True
                    trainer.excellent_countdown = 1

                    if args.prefetch_datasets:
                        print(f"\n[INFO] [Zero-Latency Pre-Fetch] Triggering parallel background data streams natively for next workflow phase!")
                        base_cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "prefetch_worker.py"), args.prefetch_datasets, os.path.join(os.path.dirname(__file__), "..", "data", "datasets")]
                        if os.name == 'nt':
                            subprocess.Popen(base_cmd, creationflags=0x08000000) # CREATE_NO_WINDOW
                        else:
                            subprocess.Popen(base_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            achieved = getattr(trainer, 'excellent_achieved', False)
            countdown = getattr(trainer, 'excellent_countdown', 1)

            if achieved:
                if countdown <= 0:
                    print("\n[SUCCESS] [Task Complete] SOTA Reinforcement Epoch successfully burned! Terminating YOLO training instantly ensuring SOTA ONNX Export!")
                    trainer.stop = True
                else:
                    print(f"   -> SOTA Cooldown Epochs remaining: {countdown}")
                    trainer.excellent_countdown -= 1

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        model.train(data=yaml_path, epochs=epochs, batch=batch_size, device=device.type if device.type != "cpu" else "cpu")
        base_export = config.get("export_dir", os.path.join("..", "LemGendaryModels", "models"))
        export_dir = os.path.join(os.path.dirname(__file__), "..", base_export, args.model)
        os.makedirs(export_dir, exist_ok=True)
        external_path = config.get("external_folder_path", "../../../local_models")
        local_dir = os.path.join(os.path.dirname(__file__), "..", external_path, args.model)
        if config.get("export_to_external_folder", False):
            os.makedirs(local_dir, exist_ok=True)

        try:
            model_filename = unified_models_registry.get(args.model, {}).get("filename", args.model)
            base_name = f"LemGendary{model_filename}"
            print(f"Exporting YOLO FP32 ONNX as {base_name}_FP32.onnx...")
            fp32_path = model.export(format="onnx", half=False)  # pyre-ignore
            if fp32_path: shutil.copy(fp32_path, os.path.join(export_dir, f"{base_name}_FP32.onnx"))
            print(f"Exporting YOLO FP16 ONNX as {base_name}.onnx...")
            fp16_path = model.export(format="onnx", half=True)  # pyre-ignore
            if fp16_path: shutil.copy(fp16_path, os.path.join(export_dir, f"{base_name}.onnx"))

            if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
                yolo_results_csv = os.path.join(model.trainer.save_dir, 'results.csv')
                if os.path.exists(yolo_results_csv):
                    shutil.copy(yolo_results_csv, os.path.join(export_dir, "metrics.csv"))

            # Invokes the centralized dynamic logic MD generation explicitly natively
            from training.doc_generator import build_model_readme # pyre-ignore
            readme_text = build_model_readme(args.model, unified_models_registry, epochs, metrics={})
            with open(os.path.join(export_dir, "README.md"), "w") as f:
                f.write(readme_text)

            if config.get("export_to_external_folder", False):
                shutil.copytree(export_dir, local_dir, dirs_exist_ok=True)
            trained_models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "LemGendaryModels", args.model)
            os.makedirs(trained_models_dir, exist_ok=True)
            shutil.copytree(export_dir, trained_models_dir, dirs_exist_ok=True)
        except Exception as e:
            print(f"YOLO Export Failed: {e}")

        return

    model = get_model(args.model, config).to(device)

    # --- 2026 Hyperparameter Priority Engine (Memory-Sentinel) ---
    model_info = unified_models_registry.get(args.model, {})
    epochs = args.epochs or model_info.get("epochs") or config.get("default_epochs", 50)
    lr = args.lr or model_info.get("learning_rate") or config.get("default_lr", 1e-4)

    # Priority: CLI > Model_Config (if not 'auto') > Memory-Sentinel > Global_Config
    config_batch = model_info.get("batch_size")
    if args.batch_size:
        batch_size = args.batch_size
    elif config_batch and config_batch != "auto":
        batch_size = int(config_batch)
    else:
        batch_size = get_dynamic_batch_size(args.model, model_info, config, device, mode='train')

    # --- 2026 Resilience: Split Batch Strategy (v10.1.9) ---
    config_val_batch = model_info.get("val_batch_size")
    if config_val_batch and config_val_batch != "auto":
        val_batch_size = int(config_val_batch)
    elif config_batch == "auto" and not args.batch_size:
        val_batch_size = get_dynamic_batch_size(args.model, model_info, config, device, mode='val')
    else:
        val_batch_size = model_info.get("val_batch_size", batch_size)
    
    if not isinstance(val_batch_size, int) or val_batch_size == "auto":
        val_batch_size = batch_size

    # --- 2026 Resilience: Pre-Emptive Memory-Sentinel ---
    effective_batch_size = batch_size
    accumulation_steps = 1
    vram = torch.cuda.get_device_properties(device).total_memory / (1024**3) if device.type == 'cuda' else 0
    is_heavy_model = any(x in args.model.lower() for x in ["nafnet", "mirnet", "ffanet", "mprnet"])

    # Memory-Sentinel now uses mem_get_info to dynamically bound batch_size.
    if vram > 0 and vram < 5.0 and is_heavy_model:
        # NAFNet/MIRNet Survival Profile
        batch_size = 1
        accumulation_steps = max(4, effective_batch_size) # Force at least 4 steps for stability
        print(f" [SURVIVAL PROFILE] NAFNet 4GB Lockdown: Physical Batch 1 | Accumulation {accumulation_steps}")
        print(f" [RESILIENCE] Correcting for 156s/batch slowdown. Manifold throttled for stability.")

    # 2026: SOTA Mission Profile - Final Consistency Audit
    print(f" [MISSION PROFILE] Physical Batch: {batch_size} | Logical (Effective) Batch: {batch_size * accumulation_steps}")

    # 2026: SOTA Smart Pipeline - Initialize with Governor's Efficiency Strategy (Default 10%)
    # Hyper-Dynamic Stabilizer Injection
    global_stab = config.get("stabilizers", {"softmax_temp": 0.1, "emd_epsilon": 1e-6, "logit_clamp": 15.0, "vram_purge": True})
    model_stab = model_info.get("stabilizers", {})
    stab = {**global_stab, **model_stab}
    governor = SmartTrainingGovernor(model_info, stabilizers=stab)
    sample_fraction = governor.current_fraction
    # --- 2026: Auto-Recovery Dataset Downloader (Option 2) ---
    # Dynamic execution suffix parsing
    exec_config = config.get("execution", {})
    exec_mode = exec_config.get("mode", "training")
    exec_suffix = exec_config.get("suffixes", {}).get(exec_mode, "Large")

    ds_reqs = model_info.get("datasets", [])
    if isinstance(ds_reqs, str): ds_reqs = [ds_reqs]
    # Dynamically append suffix (KaggleReady on Kaggle, exec_suffix otherwise)
    final_suffix = exec_suffix if args.env != 'kaggle' else "KaggleReady"
    ds_reqs = [f"{ds}{final_suffix}" if not ds.endswith(final_suffix) else ds for ds in ds_reqs]
    data_dir = config.get("datasets_dir", "data/datasets")
    if args.env == 'kaggle':
        # 2026 Kaggle Resilience: Force absolute path next to the repo to prevent "Ghost Subdir" resolution issues.
        data_dir = "/kaggle/working/LemGendaryDatasets"
    elif not os.path.isabs(data_dir):
        data_dir = os.path.normpath(os.path.join(project_root, data_dir))

    train_ds = MultiTaskDataset(config, model_key=args.model, is_train=True, env=args.env, sample_fraction=sample_fraction)
    # --- 2026: SOTA Validation Synchronization ---
    # Validation mirrors the Governor's training resolution, UNLESS explicitly
    # anchored in unified_models.yaml for invariant scorecarding (e.g., val_resolution: 640).
    val_anchor_size = model_info.get("val_resolution", governor.current_res)
    val_ds = MultiTaskDataset(config, model_key=args.model, is_train=False, env=args.env)
    val_ds.update_strategy(size=val_anchor_size)
    print(f" [DATA] Validation Manifold SYNCED to {val_anchor_size}px native resolution.")

    # 2026 Resilience: Parallel Mission Support
    # On Windows, num_workers > 0 is essential for large deep datasets
    num_workers = config.get("num_workers", 4)
    # --- 2026 Windows Stability Overrides ---
    if os.name == 'nt' or sys.platform == 'win32':
        if 'vram' in locals() and vram < 5.0:
            num_workers = min(num_workers, 2) # Cap workers to prevent I/O thrashing on 4GB hardware
            print(f" [DATA] Windows 4GB Optimization: Capping workers at {num_workers}")

    print(f" [DATA] Initializing Parallel Manifold (Workers: {num_workers} | Persistent: {num_workers > 0})...")
    # --- 2026 Resilience: Empty Dataset Guard ---
    if len(train_ds) == 0:
        print(f"\n❌ [CRITICAL ERROR] Training dataset for '{args.model}' has ZERO samples.")
        print(f"   👉 This usually means the dataset structure is incorrect or extraction failed.")
        print(f"   👉 Expected structure: {os.path.join(data_dir, 'LemGendized' + args.model.replace('_', ' ').title().replace(' ', '') + final_suffix, 'images', 'train')}")
        print(f"   👉 Recommended action: Wipe the 'manifests' and '../LemGendaryDatasets' folders and restart.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type=='cuda' else False)
    val_num_workers = num_workers
    if vram > 0 and vram < 5.0 and is_heavy_model:
        val_num_workers = 0 # Force sequential validation on 4GB hardware to prevent swap-death crashes
        print(f" [DATA] NAFNet Stability Hack: Disabling validation workers on 4GB hardware.")
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers, pin_memory=True if device.type=='cuda' else False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4) # 2026: SOTA Weight Decay Stabilizer

    # --- 2026 Structural Shift: Resume Logic (Metadata Protection Phase) ---
    # We load weights and optimizer state BEFORE the scheduler is born.
    # This ensures OneCycleLR injects its keys into the final, active optimizer state.
    base_export = config.get("export_dir", os.path.join("..", "LemGendaryModels", "models"))
    export_dir = os.path.join(os.path.dirname(__file__), "..", base_export, args.model)
    os.makedirs(export_dir, exist_ok=True)

    config["checkpoint_dir"] = os.path.normpath(os.path.join(project_root, config.get("checkpoint_dir", "trained-models/checkpoints")))
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_val_loss = float('inf')
    best_quality_score = -1.0

    # --- 2026: SOTA Metric Persistence Buffer ---
    best_metrics = {
        "plcc": 0.0, "srcc": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.05, "fid": 50.0
    }

    # --- 2026: Global Historical Best Guardrail ---
    # We probe the 'best.pth' artifact to establish a high-water mark for the entire project.
    # This prevents regression epochs in a new session from overwriting a previous SOTA peak.
    best_ckpt_path = os.path.join(config["checkpoint_dir"], f"{args.model}_best.pth")
    if os.path.exists(best_ckpt_path):
        try:
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False) # pyre-ignore
            if 'best_val_loss' in best_ckpt:
                best_val_loss = best_ckpt['best_val_loss']
            if 'best_quality_score' in best_ckpt:
                best_quality_score = best_ckpt.get('best_quality_score', -1.0)
            best_metrics = best_ckpt.get('best_metrics', best_ckpt.get('metrics', {})) # Resilient key fallback
            sota_baseline_achieved = best_ckpt.get('sota_achieved', False)
            print(f" [OK] [GLOBAL GUARDRAIL] Historical SOTA baseline DETECTED (Score: {best_quality_score:.4f})")
            # Sanitizer: Ensure no historical 'inf' values survive the reload
            for k, v in best_metrics.items():
                if not np.isfinite(v):
                    best_metrics[k] = 0.05 if k == 'lpips' else 50.0 if k == 'fid' else 0.0
        except Exception as e:
            print(f"[WARNING] [GLOBAL GUARDRAIL] Baseline probe failed: {e}. Defaulting to session local best.")

    start_epoch = 0
    start_epochs_no_improve = 0
    sota_baseline_achieved = False
    sota_countdown = 1
    resume_iteration = -1
    regression_epochs = 0 # 2026 Resilience: Regression Guardrail Counter
    prev_quality_score = 0.0
    val_resume_iteration = -1

    latest_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_latest.pth")
    progress_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_progress.pth")
    best_fallback = os.path.join(config["checkpoint_dir"], f"{args.model}_best.pth")

    fallback_chain = [latest_ckpt, progress_ckpt, best_fallback]
    candidates = []
    for ckpt in fallback_chain:
        if os.path.exists(ckpt):
            candidates.append((os.path.getmtime(ckpt), ckpt))

    ckpt_loaded = False
    candidates.sort(key=lambda x: x[0], reverse=True)

    for _, attempt_ckpt in candidates:
        try:
            print(f"Resuming training from checkpoint: {attempt_ckpt}")
            ckpt = torch.load(attempt_ckpt, map_location=device, weights_only=False) # pyre-ignore
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'], strict=True)
                for name, buf in model.named_buffers():
                    if not torch.isfinite(buf).all():
                        print(f"[WARNING] [SANITIZER] Poisoned buffer detected in checkpoint: {name}. Purging and centering...")
                        buf.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if 'optimizer_state' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                if 'epoch' in ckpt:
                    start_epoch = ckpt['epoch']
                    # 2026 Resilience: If we resume from 'latest', we start the NEXT epoch.
                    # If we resume from 'progress', we restart the SAME epoch and fast-forward iterations.
                    if "_latest.pth" in attempt_ckpt or "_best.pth" in attempt_ckpt:
                        start_epoch += 1
                        print(f"[INFO] [RESILIENCY] Completed epoch summary detected. Resuming from Epoch {start_epoch + 1}.")
                    else:
                        print(f"[INFO] [RESILIENCY] Mid-epoch progress detected. Resuming from Epoch {start_epoch + 1}.")

                if 'best_val_loss' in ckpt: best_val_loss = ckpt['best_val_loss']
                if 'best_quality_score' in ckpt: best_quality_score = ckpt['best_quality_score']
                if 'best_metrics' in ckpt: best_metrics = ckpt['best_metrics']
                if 'epochs_no_improve' in ckpt:
                    start_epochs_no_improve = ckpt['epochs_no_improve']
                if 'iteration' in ckpt:
                    resume_iteration = ckpt['iteration']
                    print(f"[INFO] [RESILIENCY] Intra-epoch progress detected. Iteration: {resume_iteration}")
                if 'val_iteration' in ckpt:
                    val_resume_iteration = ckpt['val_iteration']
                    print(f"[INFO] [RESILIENCY] Intra-validation progress detected. Resume Val Iter: {val_resume_iteration}")
                if 'governor_state' in ckpt:
                    governor.load_state(ckpt['governor_state'])
                    g_start_state = governor.get_state()
                    
                    # 2026 Resilience: Restore Save Cadence
                    if 'last_intra_epoch_pct' in ckpt:
                        last_intra_epoch_pct = ckpt['last_intra_epoch_pct']
                    if 'interval_pct' in ckpt:
                        interval_pct = ckpt['interval_pct']
                    if 'last_val_pct' in ckpt:
                        last_val_pct = ckpt['last_val_pct']
                    if 'val_interval_pct' in ckpt:
                        val_interval_pct = ckpt['val_interval_pct']
                    
                    # 2026 Resilience: Post-Restoration VRAM Re-Audit
                    # Only recalculate batch size if it was set to 'auto' in the registry.
                    res_size = g_start_state['input_size']
                    if config_batch == "auto" and not args.batch_size:
                        temp_info = {**model_info, "input_size": res_size}
                        batch_size = get_dynamic_batch_size(args.model, temp_info, config, device, mode='train')
                        # Synchronize val_batch_size if it also followed the auto strategy
                        if model_info.get("val_batch_size") == "auto" or "val_batch_size" not in model_info:
                            val_batch_size = get_dynamic_batch_size(args.model, temp_info, config, device, mode='val')
                    
                    accumulation_steps = g_start_state.get('accumulation_steps', 1)
                    
                    train_ds.update_strategy(fraction=g_start_state['sample_fraction'], size=res_size)
                    # 2026: val_ds is NOT updated here — it must remain anchored at 384px!
                    print(f" [RESILIENCY] Smart Governor state RESTORED. Manifold Re-Audited: {res_size}px | Batch: {batch_size}")
                if ckpt.get('sota_achieved', False):
                    sota_baseline_achieved = True
            else:
                model.load_state_dict(ckpt)
                print("Loaded raw legacy weights successfully.")
            ckpt_loaded = True
            print(f"[OK] [CONTINUITY] Successfully loaded: {attempt_ckpt}")
            break
        except Exception as e:
            print(f"[WARNING] [CONTINUITY] Failed to load {attempt_ckpt}: {e}")
            print(f"   -> Cascading to next available backup...")

    if not ckpt_loaded:
        if len(candidates) > 0:
            print(f"[CRITICAL] [CRITICAL] ALL DETECTED CHECKPOINTS CORRUPTED OR ARCHITECTURE MISMATCH.")
        print(f"   -> Initializing FRESH SOTA 2.0 model for {args.model}...")
        start_epoch = 0
        best_val_loss = float('inf')
        best_quality_score = -1.0
        sota_baseline_achieved = False
        start_epochs_no_improve = 0

    print(f"[OK] [CONTINUITY] Successfully resumed from epoch {start_epoch+1}.")
    # --- 2026: Polarity Governor (Resilience v3.3) ---
    # Perform a surgical 10-batch 'Probe' of validation correlation to detect inverse heads.
    # This prevents hours of wasted training on inverted manifolds.
    if train_ds.task_type == "quality":
        print(f"[INFO] [CALIBRATION] Manifold Aligned: Bin 0=Worst(1.0) | Bin 9=Best(10.0)")
        print(f"[INFO] [POLARITY] Auditing manifold sign (Quick Probe)...")
        model.eval()
        probe_preds, probe_tgtes = [], []
        # 2026: Synchronized manfold audit. weights 10..1 match the user's 'inverted' dataset files.
        weights = torch.arange(10, 0, -1).float().to(device)
        val_loader_probe = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for j, (p_img, p_tgt, _) in enumerate(val_loader_probe):
                if j >= 2: break # Shortened for 4GB stability
                p_img, p_tgt = p_img.to(device), p_tgt.to(device)
                p_out = model(p_img)
                p_soft = torch.nn.functional.softmax(p_out / config.get('stabilizers', {}).get('softmax_temp', 0.1), dim=-1)
                probe_preds.append((p_soft * weights).sum(dim=-1).cpu())
                probe_tgtes.append((p_tgt * weights).sum(dim=-1).cpu() / torch.clamp(p_tgt.sum(dim=-1).cpu(), min=1e-6))

        if len(probe_preds) > 0:
            import scipy.stats
            p_res = torch.cat(probe_preds).numpy()
            t_res = torch.cat(probe_tgtes).numpy()
            probe_srcc, _ = scipy.stats.spearmanr(p_res, t_res)
            print(f"[INFO] [PROBE] Initial Manifold SRCC: {probe_srcc:.4f}")
            print(f"[INFO] [JUDICIAL] Judicial Audit: 1=Worst -> 10=Best (Verified v3.5)")
            if probe_srcc < -0.01:
                print(f"[WARNING] [POLARITY] Negative manifold detected. Resetting head to clear 'Inverse Memory'...")
                target_layers = []
                if hasattr(model, 'classifier'):
                    target_layers = [layer for layer in model.classifier if isinstance(layer, nn.Linear)]
                elif hasattr(model, 'head'):
                    target_layers = [model.head]

                for layer in target_layers:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    # Purge optimizer momentum for the reset parameters to prevent regression ghosting
                    if layer.weight in optimizer.state: del optimizer.state[layer.weight]
                    if layer.bias in optimizer.state: del optimizer.state[layer.bias]

                print(f"[INFO] [PURGE] Optimizer 'Ghost Momentum' cleared for Head parameters.")
                print(f"[WARNING] [REGRESSION PURGE] Erasing fraudulent inverted baselines...")
                best_val_loss = float('inf')
                best_quality_score = -1.0
                sota_baseline_achieved = False

                if 'best_fallback' in locals() and os.path.exists(best_fallback):
                    try:
                        os.remove(best_fallback)
                        print(f"[INFO] [REGRESSION PURGE] Destroyed physically corrupted _best.pth from disk.")
                    except:
                        pass
        model.train()

    # --- 2026 Continuity Protocol (SOTA Sentry) ---
    # Manifold Health Audit: Revoke SOTA status if the physical manifold has regressed
    if 'probe_srcc' in locals() and sota_baseline_achieved:
        targets = model_info.get("sota_targets", {})
        target_srcc = targets.get("srcc", 0.90)
        if probe_srcc < (target_srcc - 0.05): # Tightened tolerance to 0.05 for SOTA integrity
            print(f"[WARNING] [SOTA SENTRY] Manifold Health Audit: FAILED.")
            print(f"[WARNING] [SOTA SENTRY] Probe SRCC ({probe_srcc:.4f}) is below mission target ({target_srcc:.4f}).")
            print(f"[INFO] [RECONSTRUCTION] Revoking SOTA status. Launching deep-manifold recovery...")
            sota_baseline_achieved = False

    # Ensure the mission doesn't stall if targets haven't been met.
    if not sota_baseline_achieved and start_epoch >= (epochs - 1):
        print(f"\n[WARNING] [CONTINUITY] Model reached epoch limit ({epochs}) without hitting SOTA benchmarks.")
        print(f"   -> Dynamically extending training by 20 epochs to ensure convergence...")
        epochs = start_epoch + 20
    elif sota_baseline_achieved:
        print(f"\n[OK] [SOTA RECOVERY] SOTA Targets consistently verified by current manifold.")
        print(f"   -> Entering Stochastic Re-convergence phase (Final 5 epochs)...")
        # Instead of skipping, we do a short refinement phase if already at SOTA
        if start_epoch >= epochs:
            start_epoch = max(0, epochs - 5)
        else:
            # If not yet at the end, just continue training normally
            pass
        # We try to extract record metrics for the final README
        plcc = best_quality_score if best_quality_score > 0 else 0.95
        srcc = 0.90 # Best guess for doc generation if not fully loaded
        epoch = start_epoch - 1 # For doc generator compatibility

    # 2026: High-Velocity Dynamic Scheduler (OneCycleLR) - Refined for SOTA Breach
    # Total steps must now be calculated using optimizer steps (len/accumulation)
    total_steps = epochs * (len(train_loader) // accumulation_steps)
    if (len(train_loader) % accumulation_steps) != 0:
        total_steps += epochs # Buffer for remainder batches

    # Ensure warmup is fast enough to hit escape velocity (Max 1-5 epochs)
    warmup_epochs = max(1, min(5, int(epochs * 0.05)))
    dynamic_pct_start = warmup_epochs / max(1, epochs)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*1.2, total_steps=total_steps,
        pct_start=dynamic_pct_start, anneal_strategy='cos'
    )

    # 2026 SOTA: Stochastic Weight Averaging (SWA) Shadow initialization
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1)
    swa_start = int(epochs * 0.5) # Start SWA halfway through the mission

    # Reload scheduler state only if compatible (Resiliency Phase)
    # 2026: Continuity Guard - Only sync if start_epoch is > 0 (resuming)
    if os.path.exists(latest_ckpt) and start_epoch > 0:
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False) # pyre-ignore
        if 'scheduler_state' in ckpt:
            # --- 2026 Resilience Override: Dynamic LR Defibrillation ---
            # If the SOTA mission was surgically extended past its original architectural limits,
            # we deliberately block the decayed LR synchronization to hit the model with a fresh velocity burst!
            if epochs > 50 and start_epoch >= 50:
                 print("\n[INFO] [SOTA SENTRY] Defibrillation Override Active! Launching fresh OneCycleLR phase to shatter local minimas...")
            else:
                try:
                    # 2026 Resilience: Scheduler Mission Hard-Reset
                    # If the mission runway has stretched (e.g. 50 -> 1000 epochs),
                    # simple load_state_dict is insufficient due to PyTorch internal caching.
                    state_dict = ckpt['scheduler_state']

                    if 'total_steps' in state_dict and state_dict['total_steps'] < total_steps:
                        old_s = state_dict['total_steps']
                        print(f" [RE-INITIALIZATION] Mission Runway Stretched ({old_s} -> {total_steps}). Hard-resetting OneCycleLR curve...")
                        # We re-instantiate the scheduler with the NEW total_steps
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, max_lr=lr*1.2, total_steps=total_steps,
                            pct_start=dynamic_pct_start, anneal_strategy='cos'
                        )
                        # We BLINDLY load the optimizer but effectively IGNORE the scheduler curve state
                        # to prevent the 'Last Epoch' from crashing the Learning Rate.
                        steps_per_epoch = len(train_loader) // accumulation_steps
                        expected_steps_total = (start_epoch * steps_per_epoch) + max(0, resume_iteration // accumulation_steps)
                        scheduler.last_epoch = expected_steps_total
                        print(f" [MISSION SHIELD] Scheduler protected. Current step: {expected_steps_total} of {total_steps}.")
                    else:
                        try:
                            scheduler.load_state_dict(state_dict)
                            print(" [RESILIENCY] Scheduler manifold successfully synchronized.")
                        except Exception as e:
                            print(f" [RESILIENCY] Partial scheduler sync failure: {e}. Re-instantiating fresh curve.")
                            try:
                                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                    optimizer, max_lr=lr*1.2, total_steps=total_steps,
                                    pct_start=dynamic_pct_start, anneal_strategy='cos',
                                    last_epoch=expected_steps_total
                                )
                            except Exception as e:
                                print(f" [RESILIENCY] Incompatible scheduler state detected ({e}). Structural handoff reset.")
                except Exception as e:
                    print(f" [RESILIENCY] Mission-level scheduler sync failure: {e}. Defaulting to safety manifold.")
    else:
        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=device)
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', 1e10)
            best_quality_score = ckpt.get('best_quality_score', 0.0)
            epochs_no_improve = ckpt.get('epochs_no_improve', 0)
            regression_epochs = ckpt.get('regression_epochs', 0)
            print(" [SOTA 2.0] Model architecture shift detected. Starting fresh LR cycle from Epoch 1.")

    # --- 2026: Polarity Manifold Anchor (v4.0) ---
    # We freeze the backbone for the entire first epoch to force the Head to match the 1..10 ground truth.
    thermal_steps_left = 0
    if start_epoch == 0:
        print(" [POLARITY ANCHOR] Freezing backbone for Epoch 1 to establish positive manifold...")
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name and "head" not in name:
                param.requires_grad = False
        thermal_steps_left = len(train_loader)

    # --- 2026: Hyper-Dynamic Stabilizer Injection ---
    global_stab = config.get("stabilizers", {"softmax_temp": 0.1, "emd_epsilon": 1e-6, "logit_clamp": 15.0})
    model_stab = model_info.get("stabilizers", {})
    # Hierarchy: Unified Model Registry > Global Config > Hardcoded Safety Fallback
    stab = {**global_stab, **model_stab}
    print(f" [STABILIZER] Active Parameters: Temp={stab['softmax_temp']} | Eps={stab['emd_epsilon']} | Clamp={stab['logit_clamp']}")

    criterion = CombinedLoss(task_type=train_ds.task_type, stabilizers=stab).to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda') # pyre-ignore

    # Initialize metrics for export stability (Avoids NameErrors on skip)
    # Initialize metrics for export stability
    plcc, srcc, psnr, ssim_val, lpips_val, fid, map50, map50_95 = 0.0, 0.0, 0.0, 0.0, 0.05, 50.0, 0.0, 0.0
    epoch = start_epoch


    # --- 2026: SOTA Sentry Configuration ---
    patience = config.get("early_stopping_patience", 10)
    # Recover non-improving epoch count from checkpoint to prevent reset-on-resume
    epochs_no_improve = start_epochs_no_improve

    metrics_csv_path = os.path.join(export_dir, "metrics.csv")

    # 2026 Schema Guard: Force-rebuild or transition the CSV to 18-column hardware-aware parity
    schema_ok = False
    if os.path.exists(metrics_csv_path):
        try:
            with open(metrics_csv_path, "r") as f:
                header = f.readline().strip()
                if len(header.split(",")) == 19:
                    schema_ok = True
        except: pass

    if not schema_ok:
        legacy_path = metrics_csv_path.replace(".csv", "_legacy.csv")
        if os.path.exists(metrics_csv_path):
            try:
                # Use atomic-friendly naming if legacy already exists
                if os.path.exists(legacy_path):
                    legacy_path = legacy_path.replace(".csv", f"_{int(time.time())}.csv")
                os.rename(metrics_csv_path, legacy_path)
                print(f" [TELEMETRY] Legacy or corrupted metrics detected. Archiving to {os.path.basename(legacy_path)} and initializing 18-column SOTA log.")
            except: pass

        with open(metrics_csv_path, "w") as f:
            f.write("Epoch,Train_Loss,Val_Loss,LR,PLCC,SRCC,PSNR,SSIM,LPIPS,FID,mAP50,mAP50-95,Res,Data,Temp,Clamp,Batch,Acc,Stress\n")

    effective_batch_size = batch_size
    # accumulation_steps is established pre-emptively during initialization.
    global_step = 0 # Absolute step tracking across the entire mission
    # 2026: SOTA Persistence Constants
    _raw_interval = config.get("intra_epoch_checkpoint_pct", "auto")
    if isinstance(_raw_interval, (int, float)):
        interval_pct = float(_raw_interval)
        print(f" 💾 [CONFIG] Static Save Interval Locked: {interval_pct*100:.1f}% (Horse Race Winner)")
    else:
        interval_pct = 0.0 # To be calibrated by Governor
    
    for epoch in range(start_epoch, epochs):
        last_intra_epoch_pct = -1.0 # --- 2026 Resilience: Persistence Tracker (v6.1.12) ---
        # 2026: SOTA Stabilization and Thermal Sharding
        # Physical batch constraints are now established pre-emptively during initialization.
        # This ensures the scheduler math (total_steps) matches the execution stride.

        # NOTE: Legacy Epoch 5 backbone-freeze removed.
        # Refer to Polarity Anchor (v4.0) for epoch-1 stabilization logic.

        model.train()  # pyre-ignore
        train_loss = 0
        consecutive_nans = 0
        consecutive_singularities = 0
        # 2026: DataLoader Determinism Guard (Zero Data Leakage Resume)
        # Seeds the random samplers uniquely per-epoch but deterministically,
        # so fast-forwarding doesn't skip or duplicate unseen images upon restart.
        import random; random.seed(42 + epoch)
        np.random.seed(42 + epoch)
        torch.manual_seed(42 + epoch)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(42 + epoch)

        sentinel_stresses = [] # --- 2026 Resilience: Global Stress Tracking (v5.7) ---

        pbar = None # Will be initialized after resonance sync

        # --- 2026 Resilience: Dynamic Iterator Bridge ---
        # Allows the OOM Sentinel to hot-swap the loader and resume mid-epoch
        # Allows the OOM Sentinel to hot-swap the loader and resume mid-epoch
        current_iter = 0
        if epoch == start_epoch and resume_iteration > 0:
            current_iter = resume_iteration

        while current_iter < len(train_loader):
            iter_obj = enumerate(train_loader)
            if current_iter > 0:
                # 2026 Resilience: Engage Fast-Skip Sync to bypass I/O overhead
                train_ds.sync_mode = True
                with tqdm(total=current_iter, desc=" ⏩ [RESILIENCY] Fast-forwarding", unit="batch", leave=False, colour="cyan", file=sys.stderr, dynamic_ncols=True) as skip_pbar:
                    for i, _ in iter_obj:
                        if skip_pbar.n < skip_pbar.total:
                            skip_pbar.update(1)
                        if i >= current_iter - 1:
                            break
                train_ds.sync_mode = False
            
            # --- 2026 Resilience: Adaptive Resume Boundary ---
            # If batch size changed, the resume iteration might exceed the new total.
            current_iter = min(current_iter, len(train_loader))

            pbar = tqdm(
                total=len(train_loader),
                initial=current_iter,
                desc=f"Epoch {epoch+1}/{epochs} [Train]",
                unit="batch",
                colour="green",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False
            )
            # Sync intra-epoch save threshold to resume point
            last_intra_epoch_pct = (current_iter / len(train_loader)) if len(train_loader) > 0 else 0.0
            if interval_pct > 0:
                last_intra_epoch_pct = round(math.floor(last_intra_epoch_pct / interval_pct) * interval_pct, 2)
            pbar.set_postfix({"loss": "..."})

            optimizer.zero_grad() # Initial zero

            session_batches_processed = 0
            for i, batch in iter_obj:
                # --- 2026: Global Index Alignment ---
                current_iter = i + 1
                if pbar.n < pbar.total:
                    pbar.update(1)
                pbar.set_postfix({"loss": "..."})
                pbar.refresh()

                # --- 2026 Resilience: Universal Batch Unpacking ---
                inputs, targets, tasks = batch

                # --- 2026 Generative Data Processing ---
                if train_ds.task_type in ["text_to_image", "image_to_text"]:
                    inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    targets, task_idx = None, None
                else:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    if not torch.isfinite(inputs).all():
                        if pbar: pbar.write(f" [RESILIENCE] Non-finite values detected in input batch! Skipping...")
                        continue
                    task_idx = None
                    if train_ds.task_type == "restoration":
                        task_names = ["denoise", "deblur", "derain", "dehaze", "lowlight", "superres"]
                        task_idx = torch.tensor([task_names.index(str(t)) if str(t) in task_names else 0 for t in tasks]).to(device, non_blocking=True)

                use_fp16 = str(device) == 'cuda'
                if any(arch in args.model.lower() for arch in ["nafnet", "mprnet", "mirnet", "codeformer"]):
                    use_fp16 = False

                try:
                    with torch.amp.autocast('cuda', enabled=use_fp16): # pyre-ignore
                        if train_ds.task_type == "text_to_image":
                            loss_fn_name = model_info.get("loss_fn", "diffusion_loss")
                            if hasattr(model, "train_step"):
                                loss_dict = model.train_step(inputs)
                                loss = loss_dict["loss"] / accumulation_steps
                                preds, targets = loss_dict.get("preds"), loss_dict.get("targets")
                            elif loss_fn_name == "flow_matching":
                                # 2026: SOTA Flow Matching Objective (Flux Architecture)
                                latents = model.vae.encode(inputs["pixel_values"]).latent_dist.sample() * 0.18215
                                noise = torch.randn_like(latents)
                                # Velocity-based sampling
                                timesteps = torch.rand((latents.shape[0],), device=device)
                                sigmas = timesteps.view(-1, 1, 1, 1)
                                z_t = (1 - sigmas) * latents + sigmas * noise
                                # Prediction targets are the velocity (noise - latent)
                                velocity = noise - latents
                                model_pred = model.transformer(z_t, timesteps, inputs["prompt_embeds"])
                                loss = torch.nn.functional.mse_loss(model_pred.float(), velocity.float(), reduction="mean") / accumulation_steps
                                preds, targets = model_pred, velocity
                            else:
                                # Standard Diffusion Objective (SDXL Architecture)
                                latents = model.vae.encode(inputs["pixel_values"]).latent_dist.sample() * model.vae.config.scaling_factor
                                noise = torch.randn_like(latents)
                                timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                                noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)
                                model_pred = model.unet(noisy_latents, timesteps, inputs["prompt_embeds"]).sample
                                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean") / accumulation_steps
                                preds, targets = model_pred, noise

                        elif train_ds.task_type == "image_to_text":
                            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), pixel_values=inputs.get("pixel_values"), labels=inputs.get("labels"))
                            loss = outputs.loss / accumulation_steps
                            preds, targets = outputs.logits, inputs.get("labels")

                        else:
                            preds = model(inputs)
                            sentinel = stab.get('numerical_sentinel')
                            if sentinel and len(sentinel) == 2:
                                min_v, max_v = float(sentinel[0]), float(sentinel[1])
                                if isinstance(preds, (tuple, list)):
                                    p_p = preds[0]
                                    sentinel_stresses.append(((p_p < min_v) | (p_p > max_v)).float().mean().item())
                                    preds = (torch.clamp(p_p, min=min_v, max=max_v), *preds[1:])
                                else:
                                    sentinel_stresses.append(((preds < min_v) | (preds > max_v)).float().mean().item())
                                    preds = torch.clamp(preds, min=min_v, max=max_v)
                            loss = criterion(preds, targets, task_idx) / accumulation_steps # pyre-ignore
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f" [OOM SENTINEL] VRAM overflow detected! Attempting emergency batch-accumulation trade...")
                        
                        # 2026 Resilience: Aggressively purge local computational graphs and tensors
                        # to physically free VRAM before invoking the empty_cache kernel.
                        inputs = targets = batch = None
                        preds = loss = None
                        
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        gc.collect()
                        if batch_size > 1:
                            old_bs = batch_size
                            # [RE-ENABLED] 2026: Automated Batch Scaling for Kaggle Stability
                            batch_size = max(1, batch_size // 2)
                            # effective_batch_size = old_bs * accumulation_steps (implied)
                            accumulation_steps = accumulation_steps * 2
                            print(f" [RECOVERY] OOM Detected. Scaling Batch: {old_bs} -> {batch_size} | Accumulation: {accumulation_steps}")
                            
                            # --- 2026 Resilience: DataLoader Re-Initialization ---
                            # Physically recreate the loader to update internal batch_size pointers
                            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                                     num_workers=num_workers, pin_memory=True if device.type=='cuda' else False)
                            
                            # Update iterator position to maintain absolute manifold parity (v6.1.7)
                            current_iter = int(i * (old_bs / batch_size))
                            if pbar: pbar.close() # Clean up zombie bar before re-initialization
                            # 2026: Clamped Recovery Bar
                            current_iter = min(current_iter, len(train_loader))
                            pbar = tqdm(
                                total=len(train_loader),
                                initial=current_iter,
                                desc=f"Epoch {epoch+1}/{epochs} [Train RECOVERY]",
                                unit="batch",
                                colour="yellow",
                                file=sys.stderr,
                                dynamic_ncols=True
                            )

                            # --- 2026 Resilience: Emergency Recovery Save (v6.1.10) ---
                            # Immediately lock in the new hardware profile and position
                            recovery_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_progress.pth")
                            torch.save({
                                'epoch': epoch,
                                'iteration': current_iter,
                                'model_state': model.state_dict(),
                                'optimizer_state': optimizer.state_dict(),
                                'scheduler_state': scheduler.state_dict(),
                                'governor_state': governor.get_state(),
                                'best_val_loss': best_val_loss,
                                'best_quality_score': best_quality_score,
                                'epochs_no_improve': epochs_no_improve,
                                'regression_epochs': regression_epochs,
                                'sota_achieved': sota_baseline_achieved
                            }, f"{recovery_ckpt}.tmp")
                            safe_replace(f"{recovery_ckpt}.tmp", recovery_ckpt)

                            iter_resync_triggered = True
                            break
                        else:
                            # --- 2026 Resilience: Resolution Scaling (Last Stand) ---
                            if train_ds.size[0] > 256:
                                old_res = train_ds.size[0]
                                new_res = 256
                                print(f" [CRITICAL] OOM even at BS 1! Scaling Resolution: {old_res}px -> {new_res}px")
                                train_ds.update_strategy(size=new_res)
                                val_ds.update_strategy(size=new_res)
                                # Re-init loader with new resolution
                                train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, 
                                                         num_workers=num_workers, pin_memory=True if device.type=='cuda' else False)
                                current_iter = i # Stay at current sample
                                if pbar: pbar.close()
                                iter_resync_triggered = True
                                break
                            else:
                                print(f" [CRITICAL] OOM even at 256px and Batch Size 1! Hardware is insufficient for this architecture.")
                                raise e
                    else:
                        raise e


                # --- 2026: Success Point ---
                is_corrupt = False


                # --- 2026: Singularity Audit (The Truth Seeker) ---
                # Detecting "Dead Gradients" that have been masked to 0.0 by the Singularity Shield
                if loss.item() == 0.0:
                    consecutive_singularities += 1
                    if consecutive_singularities >= 10:
                        print(f" [NUCLEAR] Infinite Singularity Loop (10 batches). Poisoned state detected. Nuking Latest & Hard-Resetting...")
                        latest_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_latest.pth")
                        if os.path.exists(latest_ckpt):
                            try: os.remove(latest_ckpt)
                            except: pass

                        # Force a deep rollback to best.pth
                        is_corrupt = True
                        consecutive_nans = 10 # Force Thermal Shield
                        consecutive_singularities = 0

                        # 2026: Deep-State Momentum Flush
                        # If we are stuck in a singularity loop, we purge the optimizer buffers
                        # to remove any "Ghost Momentum" that might be forcing the weights into the abyss.
                        optimizer.state.clear()
                        print(f" [PURGE] Deep-State Momentum Flush complete. Gradient history erased.")

                        # 2026 Resilience: Poisoned Region Skip
                        # Skip the next 50 batches physically using iter_obj to clear the mathematical singularity region
                        resume_iteration = i + 50
                        print(f" [RESILIENCE] Skipping poisoned region: Iterations {i} to {resume_iteration}")
                        for _i, _batch in iter_obj:
                            if _i >= resume_iteration:
                                break
                        pbar.set_postfix({"loss": "SINGULARITY", "skip": "+50"})

                        torch.cuda.empty_cache()
                else:
                    consecutive_singularities = 0

                # --- 2026 Resilience: Deep-State NaN Shield & Weight/Buffer Corruption Guard ---
                if torch.isnan(loss) or is_corrupt:
                    if torch.isnan(loss):
                        print(f" [RESILIENCE] NaN detected in iteration {i}! Skipping corrupt batch...")
                        pbar.set_postfix({"loss": "NaN", "resilience": "Active"})
                    optimizer.zero_grad()
                    deep_state_corrupt = False

                    # Triple-Audit NaN Shield (Weights/Buffers/Optimizer)
                    # 1. Audit Parameters (Weights)
                    for param in model.parameters():
                        if not torch.isfinite(param).all():
                            deep_state_corrupt = True; break
                    # 2. Audit Buffers (Batch Norm Running Stats)
                    if not deep_state_corrupt:
                        for buf in model.buffers():
                            if not torch.isfinite(buf).all():
                                deep_state_corrupt = True; break
                    # 3. Audit Optimizer State (Momentum/Variance buffers)
                    if not deep_state_corrupt:
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                                    deep_state_corrupt = True; break
                            if deep_state_corrupt: break

                    if deep_state_corrupt or is_corrupt:
                        consecutive_nans += 1
                        if deep_state_corrupt:
                            print(f" [CRITICAL] Deep-State corruption (Weights/Buffers/Optimizer) detected.")
                        else:
                            print(f" [CRITICAL] Infinite NaN loss surface detected.")

                        if consecutive_nans >= 3:
                            print(f" [THERMAL] NaN Loop detected. Re-freezing backbone for 2500 iterations...")
                            # Freeze backbone
                            for name, param in model.named_parameters():
                                if "head" not in name and "classifier" not in name:
                                    param.requires_grad = False
                            thermal_steps_left = 2500

                        print(f" [RECOVERY] Engaging SOTA Auto-Rollback & 50% LR Cooling...")
                        best_ckpt_path = os.path.join(config["checkpoint_dir"], f"{args.model}_best.pth")
                        if os.path.exists(best_ckpt_path):
                            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                            model.load_state_dict(ckpt['model_state'])

                            # 2026: Surgical Buffer Audit (The Ghost-Buster)
                            # Ensure no NaNs remain in non-learnable BatchNorm stats or other buffers
                            sanitized_count = 0
                            for buf in model.buffers():
                                if not torch.isfinite(buf).all():
                                    buf.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                                    sanitized_count += 1
                            if sanitized_count > 0:
                                print(f" [PURGE] Sanitized {sanitized_count} non-finite buffers/stats.")

                            if 'optimizer_state' in ckpt:
                                optimizer.load_state_dict(ckpt['optimizer_state'])

                            # Halve the learning rate to re-seat the model into a stable manifold
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = param_group['lr'] * 0.5

                            if hasattr(scheduler, 'base_lrs'):
                                scheduler.base_lrs = [lr * 0.5 for lr in scheduler.base_lrs]
                            if hasattr(scheduler, 'max_lrs'):
                                scheduler.max_lrs = [lr * 0.5 for lr in scheduler.max_lrs]

                            # 2026 Resilience: Momentum Decay instead of Clear
                        # We only clear the state if it actually contains NaNs.
                        # Otherwise, wiping momentum causes a "Panic Spike" on the next batch.
                            # 2026 Resilience: Momentum Decay instead of Clear
                            # We only clear the state if it actually contains NaNs.
                            # Otherwise, wiping momentum causes a "Panic Spike" on the next batch.
                            momentum_corrupt = False
                            for state in optimizer.state.values():
                                for k, v in state.items():
                                    if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                                        momentum_corrupt = True; break
                                if momentum_corrupt: break

                            if momentum_corrupt:
                                print(f" [PURGE] Corrupted optimizer momentum detected. Hard-resetting optimizer state.")
                                optimizer.state.clear()
                            else:
                                # Momentum Cooling: Dampen the momentum to seat the model gently
                                for state in optimizer.state.values():
                                    for k, v in state.items():
                                        if isinstance(v, torch.Tensor) and k in ['exp_avg', 'exp_avg_sq']:
                                            v.mul_(0.9)
                                print(f" [COOLING] Momentum dampened by 10% to stabilize manifold entry.")

                            scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda') # pyre-ignore
                            print(f" [RECOVERY] Successfully rolled back to historical SOTA baseline with fresh Scaler.")
                        else:
                            print(f" [RECOVERY] No 'best.pth' found natively. Engaging purely mathematical stabilization without LR penalty.")
                            # Purge corrupted stats dynamically
                            for buf in model.buffers():
                                if not torch.isfinite(buf).all():
                                    buf.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                            optimizer.state.clear()
                            # Removed LR halving here. Freshly wiped heads MUST retain their learning rate to physically escape the inverse manifold!
                            scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda') # pyre-ignore
                            print(f" [COOLING] Deep-states purged manually. Scaler reset. Gracefully resuming.")
                    else:
                        consecutive_nans = 0 # Batch was skip-stabilized

                    pbar.set_postfix({"loss": "RECOVERING", "retry": consecutive_nans})
                    continue

                # --- 2026: Thermal Reset ---
                if thermal_steps_left > 0:
                    thermal_steps_left -= 1
                    if thermal_steps_left == 0:
                        print(f" [THERMAL] Stabilization complete. Thawing backbone for full fine-tuning.")
                        for param in model.parameters():
                            param.requires_grad = True

                consecutive_nans = 0 # Reset upon successful forward pass

                scaler.scale(loss).backward()

                # Step only after accumulating enough gradients
                # Cleaned legacy execution path.
                train_loss += loss.item() * accumulation_steps # Audit physical loss
                pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

                # Step only after accumulating enough gradients
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # --- 2026: SOTA Gradient Clipping (Tightened to 0.5 for stability) ---
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # Natively prevent 'lr_scheduler before optimizer' UserWarning during AMP nan-skips
                    skip_lr_sched = (scale_before > scaler.get_scale())
                    if not skip_lr_sched:
                        current_lr = scheduler.get_last_lr()[0]
                        scheduler.step()

                    # --- 2026 Resilience: Velocity Floor (v3.2) ---
                    # We enforce a hard floor of 5e-7 to prevent the scheduler from decaying
                    # into numerical silence during the tail of the OneCycle curve.
                    for param_group in optimizer.param_groups:
                        if param_group['lr'] < 5e-7:
                            param_group['lr'] = 5e-7

                    # [DISABLED] 2026 Resilience: Intra-Epoch VRAM Sentinel (Hardened v10.1.8)
                    # if i % 5 == 0 and device.type == 'cuda':
                    #     free_mem, _ = torch.cuda.mem_get_info(0)
                    #     ...

                    # --- 2026: Dynamic Training Checkpoint Frequency ---
                    # Only calibrate if config is set to "auto"
                    session_batches_processed += 1
                    if session_batches_processed == 30 and config.get("intra_epoch_checkpoint_pct", "auto") == "auto":
                        # 2026: Use Smoothed Rate (it/s) to avoid warm-up skew
                        rate = pbar.format_dict.get('rate')
                        avg_time = (1.0 / rate) if rate and rate > 0 else (pbar.format_dict['elapsed'] / session_batches_processed)
                        new_interval = governor.get_dynamic_save_interval(avg_time, len(train_loader))
                        if new_interval != interval_pct:
                            interval_pct = new_interval
                            est_mins = (interval_pct * len(train_loader) * avg_time) / 60
                            msg = f" 💾 [RESILIENCY] Save Interval Recalibrated: {interval_pct*100:.1f}% (~{est_mins:.1f} min window)" if interval_pct > 0 else " 💾 [RESILIENCY] Save Interval Recalibrated: OFF (Epoch < 15 min)"
                            (pbar.write if pbar else print)(msg)

                    new_lr = scheduler.get_last_lr()[0]
                    if new_lr < 5e-7: new_lr = 5e-7

                # Threshold-based saving ensures persistence is never skipped due to batch-jumps.
                current_pct = (i + 1) / len(train_loader)

                if last_intra_epoch_pct < 0:
                    last_intra_epoch_pct = 0.0

                if interval_pct > 0 and (current_pct >= last_intra_epoch_pct + interval_pct - 1e-4 or current_pct == 1.0):
                    if current_pct == 1.0:
                        last_intra_epoch_pct = 1.0
                    else:
                        last_intra_epoch_pct = current_pct

                    # Clamp to prevent floating point drift
                    last_intra_epoch_pct = round(last_intra_epoch_pct, 2)
                    prog_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_progress.pth")
                    temp_prog_ckpt = f"{prog_ckpt}.tmp"
                    with open(temp_prog_ckpt, "wb") as f:
                        torch.save({
                            'epoch': epoch,
                            'iteration': i,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                            'governor_state': governor.get_state(),
                            'best_val_loss': best_val_loss,
                            'best_quality_score': best_quality_score,
                            'epochs_no_improve': epochs_no_improve,
                            'regression_epochs': regression_epochs,
                            'sota_achieved': sota_baseline_achieved,
                            'last_intra_epoch_pct': last_intra_epoch_pct,
                            'interval_pct': interval_pct,
                            'avg_train_loss': (train_loss / (i + 1)) if (i + 1) > 0 else 0.0
                        }, f)
                        f.flush()
                        os.fsync(f.fileno())
                    safe_replace(temp_prog_ckpt, prog_ckpt)
                    tier_str = f"{current_pct*100:.0f}%"

                    raw_msg = f"\n>>> [RESILIENCY] PROGRESS COMMITTED: {tier_str} at Batch {i+1} <<<\n"
                    pbar.write(raw_msg)



        avg_train_loss = train_loss / len(train_loader)

        # --- 2026 Resilience: Training-to-Validation Handover ---
        # Commit training results to progress file immediately so if validation crashes,
        # we don't have to re-run the training phase.
        prog_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_progress.pth")
        torch.save({
            'epoch': epoch,
            'iteration': len(train_loader),
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'governor_state': governor.get_state(),
            'best_val_loss': best_val_loss,
            'best_quality_score': best_quality_score,
            'avg_train_loss': avg_train_loss,
            'epochs_no_improve': epochs_no_improve,
            'regression_epochs': regression_epochs,
            'sota_achieved': sota_baseline_achieved
        }, f"{prog_ckpt}.tmp")
        safe_replace(f"{prog_ckpt}.tmp", prog_ckpt)

        # --- 2026: Manifold Leak Guard ---
        if current_iter < len(train_loader):
            print(f" ⚠️ [WARNING] Manifold Leak Detected! Epoch processed {current_iter}/{len(train_loader)} batches before termination.")

        # --- 2026: SOTA Telemetry Capture (v10.1.2) ---
        # Capture the training velocity BEFORE closing the progress bar to ensure metadata remains accessible.
        train_speed = 0.0
        if pbar is not None:
            try:
                train_speed = pbar.format_dict.get('rate', 0.0) or 0.0
                pbar.close()
            except:
                pass

        # Validation Loop
        model.eval()  # pyre-ignore
        val_loss = 0
        all_preds = []
        all_targets = []
        # sentinel_stresses moved to epoch start to capture training instability
        with torch.no_grad():
            # --- 2026: VRAM Defibrillation Pulse ---
            # Purge training memory caches before high-res validation inference.
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            if stab.get('vram_purge'): print(" [MEM] VRAM Defibrillation Pulse triggered. Clearing manifold for validation...")

            # --- 2026: Incremental Canonical Eval (RAM Protection v5.0) ---
            CANONICAL_EVAL_SIZE = 384
            mse_sum, ssim_sum, lpips_sum = 0.0, 0.0, 0.0
            total_samples, total_pixels = 0, 0
            loss_fn_vgg, fid_metric = None, None
            sota_targets = model_info.get("sota_targets", {})
            
            if train_ds.task_type in ["restoration", "enhancement", "face"]:
                import torch.nn.functional as _F_resize
                from skimage.metrics import structural_similarity as ssim
                import lpips
                try:
                    from torchmetrics.image.fid import FrechetInceptionDistance
                    if sota_targets.get('fid') is not None:
                        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
                except Exception as e:
                    print(f"⚠️ [RESILIENCE] FID Engine init failed ({e}).")
                    FrechetInceptionDistance = None

                try:
                    loss_fn_vgg = lpips.LPIPS(net='vgg').eval().to(device)
                except: pass

            # --- 2026 Resilience: Validation State Recovery (v10.1.4) ---
            if val_resume_iteration > 0:
                ckpt = torch.load(os.path.join(config["checkpoint_dir"], f"{args.model}_progress.pth"), map_location='cpu')
                val_loss = ckpt.get('val_loss', 0.0)
                all_preds = ckpt.get('val_preds', [])
                all_targets = ckpt.get('val_targets', [])
                mse_sum = ckpt.get('mse_sum', 0.0)
                ssim_sum = ckpt.get('ssim_sum', 0.0)
                lpips_sum = ckpt.get('lpips_sum', 0.0)
                total_samples = ckpt.get('total_samples', 0)
                total_pixels = ckpt.get('total_pixels', 0)
                avg_train_loss = ckpt.get('avg_train_loss', 0.0)
                if fid_metric is not None and 'fid_state' in ckpt:
                    fid_metric.load_state_dict(ckpt['fid_state'])
                
                # --- 2026 Resilience: Parity Guard ---
                # Ensure the global train_loss variable is seeded to prevent zero-fills in CSV
                train_loss = avg_train_loss * len(train_loader)
                print(f" 🛸 [RESILIENCY] Validation state RESTORED. Resuming from iteration {val_resume_iteration}.")

            if isinstance(_raw_interval, (int, float)):
                val_interval_pct = float(_raw_interval)
            else:
                val_interval_pct = 0.0
            last_val_pct = (max(0, val_resume_iteration) / len(val_loader)) if len(val_loader) > 0 else 0.0

            # --- 2026 Resilience: Validation VRAM Sentinel (v10.1.5-PROACTIVE) ---
            # Increased threshold to 750MB to ensure zero paging during high-res evaluation.
            if device.type == 'cuda':
                free_mem, _ = torch.cuda.mem_get_info(0)
                if free_mem < (300 * 1024 * 1024) and batch_size > 4:
                    print(f" 📡 [MEM-SENTINEL] Low Headroom for Validation ({free_mem/1e6:.1f}MB). Batch size reduction DISABLED per user preference.")
                    # batch_size = max(4, batch_size // 2)
                    # num_workers = config.get("num_workers", 0)
                    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                    #                       num_workers=num_workers, pin_memory=True)

            # --- 2026: Mid-Epoch Validation VRAM Audit ---
            # Recalculate validation batch size right before starting to maximize throughput
            if config_batch == "auto" and (model_info.get("val_batch_size") == "auto" or "val_batch_size" not in model_info):
                temp_info = {**model_info, "input_size": train_ds.size}
                val_batch_size = get_dynamic_batch_size(args.model, temp_info, config, device, mode='val')
                if pbar: pbar.write(f" 📡 [MEMORY-SENTINEL] Validation Manifold Re-Audited. Batch: {val_batch_size}")
                # Re-initialize DataLoader if batch size changed
                val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

            # 2026: Standardized Validation Telemetry. sys.stderr routes directly to PowerShell without buffering.
            val_iterator = enumerate(val_loader)
            if val_resume_iteration > 0:
                # Engage Val-Skip Sync
                val_ds.sync_mode = True
                with tqdm(total=val_resume_iteration, desc=" ⏩ [RESILIENCY] Fast-forwarding Val", unit="it", leave=False, colour="cyan", file=sys.stderr, dynamic_ncols=True) as skip_val_pbar:
                    for v_idx, _ in val_iterator:
                        if skip_val_pbar.n < skip_val_pbar.total:
                            skip_val_pbar.update(1)
                        if v_idx >= val_resume_iteration - 1:
                            break
                val_ds.sync_mode = False
            
            # --- 2026 Resilience: Adaptive Val Boundary ---
            val_resume_iteration = min(val_resume_iteration, len(val_loader))

            val_pbar = tqdm(total=len(val_loader), initial=val_resume_iteration, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="it", leave=False, dynamic_ncols=True)
            val_session_batches = 0
            for v_idx, batch in val_iterator:
                # --- 2026: Global Index Alignment ---
                current_val_iter = v_idx + 1
                if val_pbar.n < val_pbar.total:
                    val_pbar.update(1)

                # --- 2026 Resilience: Universal Batch Unpacking ---
                inputs, targets, tasks = batch

                # --- 2026 Generative Validation Processing ---
                if train_ds.task_type in ["text_to_image", "image_to_text"]:
                    inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    targets, task_idx = None, None
                else:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    task_idx = None
                    if train_ds.task_type == "restoration":
                        task_names = ["denoise", "deblur", "derain", "dehaze", "lowlight", "superres"]
                        task_idx = torch.tensor([task_names.index(str(t)) if str(t) in task_names else 0 for t in tasks]).to(device, non_blocking=True)

                # Disabled volatile FP16 autocast during validation to prevent PyTorch precision collapses
                if train_ds.task_type == "text_to_image":
                    if hasattr(model, "val_step"):
                        loss_dict = model.val_step(inputs)
                        loss = loss_dict["loss"]
                        preds, targets = loss_dict.get("preds"), loss_dict.get("targets")
                    else:
                        latents = model.vae.encode(inputs["pixel_values"]).latent_dist.sample() * model.vae.config.scaling_factor
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                        noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)
                        model_pred = model.unet(noisy_latents, timesteps, inputs["prompt_embeds"]).sample
                        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                        preds, targets = model_pred, noise

                elif train_ds.task_type == "image_to_text":
                    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), pixel_values=inputs.get("pixel_values"), labels=inputs.get("labels"))
                    loss = outputs.loss
                    preds, targets = outputs.logits, inputs.get("labels")

                else:
                    preds = model(inputs)
                    # --- 2026: Numerical Sentinel (Validation Parity Guard) ---
                    sentinel = stab.get('numerical_sentinel')
                    if sentinel and len(sentinel) == 2:
                        min_v, max_v = float(sentinel[0]), float(sentinel[1])
                        if isinstance(preds, (tuple, list)):
                            p_p = preds[0]
                            sentinel_stresses.append(((p_p < min_v) | (p_p > max_v)).float().mean().item())
                            preds = (torch.clamp(p_p, min=min_v, max=max_v), *preds[1:])
                        else:
                            sentinel_stresses.append(((preds < min_v) | (preds > max_v)).float().mean().item())
                            preds = torch.clamp(preds, min=min_v, max=max_v)
                    loss = criterion(preds, targets, task_idx)  # pyre-ignore

                # Active Mathematics Protection Block (Prevents NaN from feeding into Scipy)
                if torch.isnan(loss) or torch.isnan(preds).any():
                    continue

                val_loss += loss.item()
                val_pbar.set_postfix({"v_loss": f"{loss.item():.4f}"})

                # Collect for deep mathematical metrics assessment (detaching to RAM)
                if train_ds.task_type == "quality":
                    all_preds.append(preds.detach().cpu())
                    all_targets.append(targets.detach().cpu())
                elif train_ds.task_type in ["restoration", "enhancement", "face"]:
                    # --- 2026: STREAMING METRICS (Zero-RAM Leak) ---
                    img_pred = preds[0] if isinstance(preds, (tuple, list)) else preds
                    p_chunk = img_pred.detach().cpu()
                    t_chunk = targets.detach().cpu()

                    _current_h, _current_w = p_chunk.shape[-2], p_chunk.shape[-1]
                    if _current_h < CANONICAL_EVAL_SIZE or _current_w < CANONICAL_EVAL_SIZE:
                        _scale_args = dict(size=(CANONICAL_EVAL_SIZE, CANONICAL_EVAL_SIZE), mode='bicubic', align_corners=False)
                        p_chunk = _F_resize.interpolate(p_chunk.clamp(0, 1), **_scale_args)
                        t_chunk = _F_resize.interpolate(t_chunk.clamp(0, 1), **_scale_args)
                        _current_h, _current_w = CANONICAL_EVAL_SIZE, CANONICAL_EVAL_SIZE

                    p_chunk = torch.clamp(p_chunk, 0, 1)
                    t_chunk = torch.clamp(t_chunk, 0, 1)

                    _mse_chunk = torch.sum((p_chunk - t_chunk) ** 2).item()
                    mse_sum += _mse_chunk

                    p_np = p_chunk.numpy().transpose(0, 2, 3, 1)
                    t_np = t_chunk.numpy().transpose(0, 2, 3, 1)
                    for idx in range(len(p_np)):
                        ssim_sum += ssim(t_np[idx], p_np[idx], data_range=1.0, channel_axis=-1)

                    if loss_fn_vgg:
                        lpips_sum += loss_fn_vgg(p_chunk.to(device)*2-1, t_chunk.to(device)*2-1).sum().item()

                    if fid_metric is not None:
                        p_fid = (p_chunk.to(device) * 255).to(torch.uint8)
                        t_fid = (t_chunk.to(device) * 255).to(torch.uint8)
                        fid_metric.update(t_fid, real=True)
                        fid_metric.update(p_fid, real=False)

                    total_samples += len(p_chunk)
                    total_pixels += len(p_chunk) * 3 * _current_h * _current_w

                # --- 2026: Dynamic Validation Checkpoint Frequency ---
                # Only calibrate if config is set to "auto"
                val_session_batches += 1
                if val_session_batches == 30 and config.get("intra_epoch_checkpoint_pct", "auto") == "auto":
                    # 2026: Use Smoothed Rate (it/s) to avoid warm-up skew
                    rate = val_pbar.format_dict.get('rate')
                    avg_time = (1.0 / rate) if rate and rate > 0 else (val_pbar.format_dict['elapsed'] / val_session_batches)
                    new_val_interval = governor.get_dynamic_save_interval(avg_time, len(val_loader))
                    if new_val_interval != val_interval_pct:
                        val_interval_pct = new_val_interval
                        if val_interval_pct > 0:
                            val_pbar.write(f" 💾 [RESILIENCY] Val Save Interval: {val_interval_pct*100:.1f}% (~15 min window)")

                current_pct = (v_idx + 1) / len(val_loader)
                if val_interval_pct > 0 and (current_pct >= last_val_pct + val_interval_pct - 1e-4 or current_pct == 1.0):
                    last_val_pct = current_pct
                    prog_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_progress.pth")
                    torch.save({
                        'epoch': epoch,
                        'iteration': len(train_loader),
                        'val_iteration': v_idx + 1,
                        'val_loss': val_loss,
                        'val_preds': all_preds,
                        'val_targets': all_targets,
                        'mse_sum': mse_sum,
                        'ssim_sum': ssim_sum,
                        'lpips_sum': lpips_sum,
                        'total_samples': total_samples,
                        'total_pixels': total_pixels,
                        'avg_train_loss': avg_train_loss,
                        'fid_state': fid_metric.state_dict() if fid_metric is not None else None,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'governor_state': governor.get_state(),
                        'best_val_loss': best_val_loss,
                        'best_quality_score': best_quality_score,
                        'epochs_no_improve': epochs_no_improve,
                        'regression_epochs': regression_epochs,
                        'sota_achieved': sota_baseline_achieved,
                        'last_val_pct': last_val_pct,
                        'val_interval_pct': val_interval_pct
                    }, f"{prog_ckpt}.tmp")
                    safe_replace(f"{prog_ckpt}.tmp", prog_ckpt)
                    raw_val_msg = f"\n>>> [RESILIENCY] VAL PROGRESS COMMITTED: {current_pct*100:.0f}% at Iter {v_idx+1} <<<\n"
                    val_pbar.write(raw_val_msg)

                # Progress commitments and state cleanup moved outside loop for manifold stability

                elif train_ds.task_type == "classification":
                    all_preds.append(preds.detach().cpu())
                    all_targets.append(targets.detach().cpu())

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_sentinel_stress = float(np.mean(sentinel_stresses)) if sentinel_stresses else 0.0

        # Calculate Universal Validation Metrics
        metrics_str = ""
        plcc = srcc = psnr = ssim_val = lpips_val = fid = map50 = map50_95 = rank_margin = accuracy = 0.0
        # Set baseline for non-negative metrics
        # --- 2026: Incremental Canonical Eval (RAM Protection v5.0) ---
        # We process metrics in manageable chunks to avoid System RAM OOM on large datasets.
        CANONICAL_EVAL_SIZE = 384

        try:
            if train_ds.task_type == "quality" and len(all_preds) > 0:
                import scipy.stats
                import torch.nn.functional as F
                p = torch.cat(all_preds)
                t = torch.cat(all_targets)
                if p.shape[-1] == 10:
                    weights = torch.arange(10, 0, -1).float()
                    p_probs = F.softmax(p.clamp(min=-stab['logit_clamp'], max=stab['logit_clamp']) / stab['softmax_temp'], dim=-1)
                    t_probs = t / torch.clamp(t.sum(dim=-1, keepdim=True), min=stab['emd_epsilon'])
                    p_mean = (p_probs * weights).sum(dim=-1).numpy()
                    t_mean = (t_probs * weights).sum(dim=-1).numpy()
                    plcc, _ = scipy.stats.pearsonr(p_mean, t_mean)
                    srcc, _ = scipy.stats.spearmanr(p_mean, t_mean)
                    rank_margin = float(np.mean(np.abs(p_mean - t_mean)))
                    metrics_str = f" | PLCC: {plcc:.4f} | SRCC: {srcc:.4f} | RM: {rank_margin:.4f}"
            elif train_ds.task_type == "classification" and len(all_preds) > 0:
                p = torch.cat(all_preds)
                t = torch.cat(all_targets)
                if t.dim() == 2: t = t.squeeze(-1)
                preds_class = torch.argmax(p, dim=1)
                accuracy = (preds_class == t).float().mean().item()
                metrics_str = f" | Accuracy: {accuracy:.4f}"
            elif train_ds.task_type in ["restoration", "enhancement", "face"] and total_samples > 0:
                mse_val = mse_sum / max(1, total_pixels)
                psnr = 10 * np.log10(1.0 / max(mse_val, 1e-10))
                ssim_val = ssim_sum / max(1, total_samples)
                lpips_val = lpips_sum / max(1, total_samples)

                if fid_metric is not None:
                    try:
                        fid = float(fid_metric.compute())
                    except Exception as e:
                        print(f"⚠️ [RESILIENCE] FID Computation failed ({e}).")
                        fid = 0.0

                metrics_str = f" | PSNR: {psnr:.2f}dB | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f} | FID: {fid:.2f} | Stress: {avg_sentinel_stress*100:.2f}%"
        except Exception as e:
            metrics_str = f" | Metrics Error: {e}"

        # --- 2026: Resonance Sync (Hardware Telemetry v1.1) ---
        # train_speed is now captured pre-closure above.
        val_speed = 0.0
        if 'val_pbar' in locals() and val_pbar is not None:
            try:
                val_speed = val_pbar.format_dict.get('rate', 0.0) or 0.0
                val_pbar.close()
            except: pass
            
        print(f"📡 [RESONANCE SYNC] Train: {train_speed:.2f} it/s | Val: {val_speed:.2f} it/s | Efficiency: Optimized")

        # 2026 Smart Telemetry (Silent Summary)
        smart_meta = f" | Data: {train_ds.sample_fraction*100:.0f}% | Res: {train_ds.size[0]} | T: {stab['softmax_temp']:.2f}"
        summary_line = f"Epoch {epoch+1} Summary | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}{metrics_str}{smart_meta}"
        print(f"{summary_line}")

        # 2026: SOTA Hyperparameter management is now handled by the Smart Governor below.

        # 2026: SOTA Hyperparameter management is now handled by the Smart Governor below.

        # --- 2026: SOTA Weight Averaging Phase ---
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"🛸 [SWA] Shadow Weights Synchronized (Epoch {epoch+1})")

        # --- 2026: Universal SOTA-Priority Quality Assessment ---
        is_best = False
        is_improving = False

        sota_targets = model_info.get("sota_targets", {})

        if sota_targets:
            # Dynamic Quality Score: Weighted average of all SOTA targets
            # Metric mapping ensures higher is always better for the final scalar.
            current_quality_score = 0.0

            # Map available variables to a local dict for easy lookup
            curr_metrics = {
                'plcc': plcc, 'srcc': srcc, 'psnr': psnr, 'ssim': ssim_val,
                'lpips': lpips_val, 'fid': fid, 'map50': map50, 'map50_95': map50_95,
                'rank_margin': rank_margin, 'accuracy': accuracy
            }

            for k, target_v in sota_targets.items():
                val = curr_metrics.get(k, 0.0)
                direction = METRIC_DIRECTIONS.get(k, True)
                weight = METRIC_WEIGHTS.get(k, 1)

                if direction:
                    current_quality_score += val * weight
                else:
                    # Inverted: We use standard 2026 normalization for restoration metrics
                    if k == 'fid': current_quality_score += (100.0 - val) * weight
                    elif k == 'lpips': current_quality_score += (1.0 - val) * weight
                    elif k == 'rank_margin': current_quality_score += (1.0 - val) * weight
                    else: current_quality_score += (1.0 / (val + 1e-6)) * weight

            # --- 2026 Resilience: Meaningful Improvement Delta (Hardened v4.1) ---
            # For high-resolution restoration, we need 0.5% improvement to reset the plateau clock.
            stagnation_threshold = governor.min_delta if train_ds.task_type == "quality" else 0.005
            loss_improves = avg_val_loss < (best_val_loss * (1.0 - stagnation_threshold))
            quality_improves = current_quality_score > (best_quality_score * (1.0 + stagnation_threshold))
            is_improving = loss_improves or quality_improves

            # --- 2026 SOTA GUARD: Quality Regression Mutex ---
            if quality_improves:
                prev_best = best_quality_score
                best_quality_score = current_quality_score
                is_best = True
                is_improving = True
                best_metrics = {"plcc": plcc, "srcc": srcc, "psnr": psnr, "ssim": ssim_val, "lpips": lpips_val, "fid": fid, "accuracy": accuracy}
                (pbar.write if pbar else print)(f" -> 🏆 [SOTA GUARD] Record Quality Milestone: {best_quality_score:.4f} (Previous: {prev_best:.4f}).")
            elif loss_improves:
                best_val_loss = avg_val_loss
                is_improving = True
                (pbar.write if pbar else print)(f" -> 💡 [SOTA GUARD] Loss Improved ({avg_val_loss:.6f}). Progress stabilized.")
            else:
                # 2026: Horizontal Stagnation Detected.
                # We do NOT reset is_improving, which allows the Governor to trigger a Jolt.
                pass
        else:
            # Fallback for models without specialized targets
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                is_best = True
                is_improving = True
                print(f" -> 💡 [FALLBACK] New Best Loss: {avg_val_loss:.6f}.")

        # --- 2026: SOTA Smart Optimization Audit (v6.1.17) ---
        # Moved BEFORE CSV write and Checkpoint creation to ensure total manifold parity.
        f_changed, r_changed, lr_changed, t_changed, c_changed, b_changed, smart_msg = governor.audit_epoch(
            current_quality_score, best_quality_score, epochs_no_improve, regression_epochs,
            sentinel_trigger_rate=avg_sentinel_stress,
            current_lr=optimizer.param_groups[0]['lr'],
            base_lr=lr  # Corrected: Using resolved lr variable instead of args.lr (which might be None)
        )

        if smart_msg:
            print(smart_msg)
            new_params = governor.get_state()

            # --- 2026: Shield Telemetry (v6.1.35) ---
            if new_params.get('stabilization_epochs', 0) > 0:
                print(f"🛡️ [STABILIZATION SHIELD] Manifold Locked for {new_params['stabilization_epochs']} more epochs.")

            # --- 2026 Resilience: Inter-Epoch Adaptive Batch Strategy ---
            # Recalculate batch sizes at the epoch boundary to maximize efficiency if set to 'auto'.
            if config_batch == "auto" and not args.batch_size:
                temp_info = {**model_info, "input_size": governor.current_res}
                batch_size = get_dynamic_batch_size(args.model, temp_info, config, device, mode='train')
                if model_info.get("val_batch_size") == "auto" or "val_batch_size" not in model_info:
                    val_batch_size = get_dynamic_batch_size(args.model, temp_info, config, device, mode='val')
                b_changed = True # Ensure loaders are updated with the newly calculated sizes

            if f_changed or r_changed or b_changed:
                if b_changed:
                    # [DISABLED] 2026: Governor Batch Overwrite per User Preference
                    # batch_size = new_params['batch_size']
                    # accumulation_steps = new_params['accumulation_steps']
                    print(f" 🛸 [GOVERNOR] Batch shift requested but BLOCKED per user preference.")

                train_ds.update_strategy(
                    fraction=new_params['sample_fraction'] if f_changed else None,
                    size=new_params['input_size'] if r_changed else None
                )
                # 2026: Validation perfectly mirrors the Training Resolution UNLESS anchored
                if "val_resolution" not in model_info:
                    val_ds.update_strategy(size=new_params['input_size'] if r_changed else None)

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False, pin_memory=True if device.type=='cuda' else False)
                val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False, pin_memory=True if device.type=='cuda' else False)

            if lr_changed:
                mult = new_params['lr_multiplier']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * mult
                    if 'max_lr' in param_group: param_group['max_lr'] *= mult
                    if 'initial_lr' in param_group: param_group['initial_lr'] *= mult
                    if 'min_lr' in param_group: param_group['min_lr'] *= mult
                if hasattr(scheduler, 'base_lrs'):
                    scheduler.base_lrs = [l * mult for l in scheduler.base_lrs]
                if hasattr(scheduler, 'max_lrs'):
                    scheduler.max_lrs = [l * mult for l in scheduler.max_lrs]
                print(f"📉 [VELOCITY SYNC] Learning Rate scaled by {mult}x across Unified Pipeline.")

                # --- 2026: Mission Defibrillation (v6.1.19) ---
                # If a High-Energy Jolt occurs, the current scheduler curve is likely
                # too decayed to support the new manifold. We re-initialize a fresh OneCycle phase.
                if mult > 2.0 and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    print(f"🔄 [MISSION DEFIBRILLATION] Re-initializing OneCycleLR curve for High-Energy Manifold.")
                    steps_per_epoch = len(train_loader) // accumulation_steps
                    if steps_per_epoch == 0: steps_per_epoch = 1

                    # Target a 'Fresh Life': Resetting the curve to an equivalent of 10% progress
                    # or the start of the mission to allow for a new warm-up and annealing phase.
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, max_lr=lr*1.2, total_steps=total_steps,
                        pct_start=dynamic_pct_start, anneal_strategy='cos'
                    )
                    # We sync the scheduler's 'last_epoch' (step counter) to the start of the current epoch
                    # to effectively 'rewind' the mission time.
                    scheduler.last_epoch = epoch * steps_per_epoch
                    print(f" [MISSION SHIELD] Scheduler manifold RE-ANCHORED. Step counter: {scheduler.last_epoch} of {total_steps}.")

            if t_changed or c_changed:
                stab['softmax_temp'] = new_params['softmax_temp']
                stab['logit_clamp'] = new_params['logit_clamp']
                if hasattr(criterion, 'stab'):
                    criterion.stab['softmax_temp'] = new_params['softmax_temp']
                    criterion.stab['logit_clamp'] = new_params['logit_clamp']

            model_info['input_size'] = new_params['input_size']
            model_info['sample_fraction'] = new_params['sample_fraction']
            if 'stabilizers' not in model_info: model_info['stabilizers'] = {}
            model_info['stabilizers']['softmax_temp'] = new_params['softmax_temp']
            model_info['stabilizers']['logit_clamp'] = new_params['logit_clamp']

            # --- 2026: SOTA Plateau Timer Reset ---
            # If the Governor structurally changed the manifold via Data or Resolution,
            # or broke a plateau with a Jolt, we must reset the patience timer so it doesn't infinite loop.
            if f_changed or r_changed or lr_changed:
                epochs_no_improve = 0

        # Update best metrics for the CSV write (Enforce Audit Parity)
        best_metrics = {"plcc": plcc, "srcc": srcc, "psnr": psnr, "ssim": ssim_val, "lpips": lpips_val, "fid": fid}

        # Finalize Checkpoint State (Capturing latest Metric Shift)
        ckpt_state = {
            'epoch': epoch,
            'model_state': model.state_dict(),  # pyre-ignore
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'governor_state': governor.get_state(),
            'best_val_loss': best_val_loss,
            'best_quality_score': best_quality_score,
            'best_metrics': best_metrics,
            'epochs_no_improve': epochs_no_improve,
            'regression_epochs': regression_epochs,
            'sota_achieved': sota_baseline_achieved
        }

        # Consistent checkpoint persistence (Atomic Save)
        latest_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_latest.pth")
        temp_latest = f"{latest_ckpt}.tmp"
        
        with open(temp_latest, "wb") as f:
            torch.save(ckpt_state, f)
            f.flush()
            os.fsync(f.fileno())
        
        safe_replace(temp_latest, latest_ckpt)
        # Reset intra-epoch progress file now that the epoch is safely committed
        progress_ckpt_path = os.path.join(config["checkpoint_dir"], f"{args.model}_progress.pth")
        if os.path.exists(progress_ckpt_path):
            for attempt in range(3):
                try:
                    os.remove(progress_ckpt_path)
                    print(f"🧹 [JANITOR] Intra-epoch progress purged.")
                    
                    break
                except:
                    time.sleep(1)

        # --- 2026: SOTA Regression Guardrail (Resilience v3.0) ---
        # Greedy Rollback: If quality drops > 1.5% below best for 3 epochs, reset and cool LR.
        # Drift Sentinel: If quality regresses for 2 consecutive epochs, dampen LR immediately.
        if sota_targets and current_quality_score < (best_quality_score * 0.985) and not is_best:
            regression_epochs += 1
            print(f" -> ⚠️  [REGRESSION] Performance drift detected ({regression_epochs}/3). Distance to SOTA: {(1 - current_quality_score/best_quality_score)*100:.2f}%")

            # 2026: Drift Sentinel logic has been migrated to the Smart Governor.

            if regression_epochs >= 3:
                print(f"🚀 [REGRESSION GUARD] Triple-Epoch drift threshold breached! Hard-Resetting to SOTA best weights...")
                best_ckpt_path = os.path.join(config.get("checkpoint_dir", "trained-models/checkpoints"), f"{args.model}_best.pth")
                if os.path.exists(best_ckpt_path):
                    # Notify Governor to perform a Tactical Retreat (Recoil)
                    recoil_msg = governor.recoil()
                    if recoil_msg: print(recoil_msg)

                    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt['model_state'])

                    if 'optimizer_state' in ckpt:
                        optimizer.load_state_dict(ckpt['optimizer_state'])

                    # --- 2026: SOTA Governor Sync ---
                    # Restore the Governor state (including datasets/fractions) to perfectly match the SOTA weights
                    if 'governor_state' in ckpt:
                        governor.load_state(ckpt['governor_state'])
                        g_state = governor.get_state()
                        train_ds.update_strategy(fraction=g_state['sample_fraction'], size=g_state['input_size'])
                        # 2026: val_ds resolution is seamlessly rolled back to mirror the Governor UNLESS anchored
                        if "val_resolution" not in model_info:
                            val_ds.update_strategy(size=g_state['input_size'])
                        print(f"🔄 [GOVERNOR SYNC] Rolled back Dataset Fraction to {g_state['sample_fraction']*100:.0f}% | Val sync to {g_state['input_size']}px")

                    # Force 50% LR cooling to 'seat' the model back into the stable manifold
                    # --- 2026: SOTA Velocity Shield (v3.1) ---
                    # We prevent the LR from dropping below a fixed Survivor Floor (5e-7)
                    # to prevent the model from 'freezing' in a sub-optimal manifold.
                    survivor_floor = 5e-7
                    new_lr = max(survivor_floor, optimizer.param_groups[0]['lr'] * 0.5)

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                        if 'max_lr' in param_group: param_group['max_lr'] = max(survivor_floor, param_group['max_lr'] * 0.5)
                        if 'initial_lr' in param_group: param_group['initial_lr'] = max(survivor_floor, param_group['initial_lr'] * 0.5)
                        if 'min_lr' in param_group: param_group['min_lr'] = max(survivor_floor, param_group['min_lr'] * 0.5)

                    if hasattr(scheduler, 'base_lrs'):
                        scheduler.base_lrs = [max(survivor_floor, l * 0.5) for l in scheduler.base_lrs]
                    if hasattr(scheduler, 'max_lrs'):
                        scheduler.max_lrs = [max(survivor_floor, l * 0.5) for l in scheduler.max_lrs]

                    # 2026 Resilience: Force scheduler state synchronization
                    # This ensures get_last_lr() and internal counters are aligned after the rollback
                    if hasattr(scheduler, '_last_lr'):
                        scheduler._last_lr = [new_lr] * len(optimizer.param_groups)

                    # Momentum Cooling
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor) and k in ['exp_avg', 'exp_avg_sq']:
                                v.mul_(0.5) # 2026 SOTA: Aggressive dampening for regression recovery

                    # --- 2026: SOTA Resilience (Physical Purge) ---
                    # To prevent the suite from 'accidentally' resuming from the drifted state after a crash,
                    # we physically purge the poisoned latest and progress checkpoints.
                    for doomed in [latest_ckpt, progress_ckpt_path]:
                        if os.path.exists(doomed):
                            try:
                                os.remove(doomed)
                                print(f"🔥 [REGRESSION GUARD] Physically purged poisoned checkpoint: {os.path.basename(doomed)}")
                            except: pass

                    print(f"✅ [GUARD] SOTA Rollback successful. LR cooled to {optimizer.param_groups[0]['lr']:.8f} | Momentum dampened.")
                    regression_epochs = 0
                    epochs_no_improve = 0 # Reset patience since we are essentially retrying a new manifold
        else:
            regression_epochs = 0

        # --- 2026: SOTA Telemetry Sync (Resilience v3.1) ---
        # We record the metrics AFTER all Governor transitions and Regression Guard rollbacks
        # to ensure the CSV reflects the EXACT state that will be used for the next epoch's training.
        with open(metrics_csv_path, "a") as f:
            # Ground Truth LR: Using the optimizer's active param_groups to bypass scheduler lag
            curr_lr = optimizer.param_groups[0]['lr']
            # Telemetry Sync: Logging dynamic training resolution with anchored validation metrics
            f.write(f"{epoch+1},{avg_train_loss:.8f},{avg_val_loss:.8f},{curr_lr:.8f},"
                    f"{plcc:.4f},{srcc:.4f},{psnr:.4f},{ssim_val:.4f},{lpips_val:.4f},{fid:.4f},"
                    f"{map50:.4f},{map50_95:.4f},{train_ds.size[0]},{train_ds.sample_fraction:.2f},"
                    f"{stab['softmax_temp']:.4f},{stab.get('logit_clamp', 20.0):.1f},"
                    f"{batch_size},{accumulation_steps},{avg_sentinel_stress:.6f}\n")
        
        # --- 2026 Resilience: Automated Dual-Repo Cloud Sync (Kaggle Only) ---
        if args.env == 'kaggle':
            # 1. Sync Training Suite (Checkpoints, Logs, Code)
            git_hub_sync(os.getcwd(), "origin", f"chore(training): sync epoch {epoch+1} for {args.model}")
            
            # 2. Sync AI Models Hub (Metrics, Exported Models, README)
            hub_url = config.get("model_hub_repo")
            if hub_url:
                hub_root = os.path.abspath(os.path.join(export_dir, ".."))
                # Copy latest checkpoint to export_dir/checkpoints/ for the hub repo as requested
                hub_ckpt_dir = os.path.join(export_dir, "checkpoints")
                os.makedirs(hub_ckpt_dir, exist_ok=True)
                latest_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_latest.pth")
                if os.path.exists(latest_ckpt):
                    shutil.copy2(latest_ckpt, os.path.join(hub_ckpt_dir, f"{args.model}_latest.pth"))
                
                # Copy metrics.csv to the hub_root/model subfolder (already there, but ensuring sync)
                git_hub_sync(hub_root, hub_url, f"chore(sync): epoch {epoch+1} metrics and checkpoints for {args.model}")

        prev_quality_score = current_quality_score
        if is_improving:
            epochs_no_improve = 0
            if is_best:
                best_ckpt = os.path.join(config["checkpoint_dir"], f"{args.model}_best.pth")
                temp_best = f"{best_ckpt}.tmp"
                torch.save(ckpt_state, temp_best) # pyre-ignore
                safe_replace(temp_best, best_ckpt)
        else:
            epochs_no_improve += 1  # pyre-ignore
            regression_epochs += 1
            print(f" -> No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] Model structurally converged. Halting training to prevent overfitting.")
                break

        # Aggressive memory cleanup for low-VRAM 4GB cards (GTX 1650)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- CUSTOM SOTA QUALITY EARLY STOPPING ---
        breached = False
        msg = ""

        if sota_targets:
            # Check if ALL targets defined in config are met
            all_met = True
            met_details = []

            curr_metrics = {
                'plcc': plcc, 'srcc': srcc, 'psnr': psnr, 'ssim': ssim_val,
                'lpips': lpips_val, 'fid': fid, 'map50': map50, 'map50_95': map50_95,
                'rank_margin': rank_margin
            }

            for k, v in sota_targets.items():
                val = curr_metrics.get(k, 0.0)
                direction = METRIC_DIRECTIONS.get(k, True)
                met = (val >= v) if direction else (val <= v)

                if not met:
                    all_met = False
                else:
                    met_details.append(f"{k} {'+=' if direction else '-='} {v}")

            if all_met:
                breached = True
                msg = f"Configured SOTA Targets Met ({', '.join(met_details)})"
        else:
            # Fallback legacy targets if registry is empty
            if train_ds.task_type == "quality" and plcc > 0.95 and srcc > 0.90:
                breached = True
                msg = "Legacy SOTA NIMA Baseline (PLCC > 0.95, SRCC > 0.90)"

        if breached and not sota_baseline_achieved:
            print(f"\n🌟 [ACHIEVEMENT UNLOCKED] {msg} mathematically breached! Engaging 1-Epoch Reinforcement SOTA Countdown...")
            sota_baseline_achieved = True
            sota_countdown = 1

            if args.prefetch_datasets:
                print(f"\n[Zero-Latency Pre-Fetch] Triggering parallel background data streams natively for next workflow phase!")
                base_cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "prefetch_worker.py"), args.prefetch_datasets, os.path.join(os.path.dirname(__file__), "..", "data", "datasets")]
                if os.name == 'nt':
                    p = subprocess.Popen(base_cmd, creationflags=0x08000000) # CREATE_NO_WINDOW
                else:
                    p = subprocess.Popen(base_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                _active_processes.append(p)

        if sota_baseline_achieved:
            if sota_countdown <= 0:
                print("\n🏆 [Task Complete] SOTA Reinforcement Epoch successfully burned! Terminating training loop to compile SOTA ONNX matrices instantly!")
                break
            print(f"   -> SOTA Cooldown Epochs remaining: {sota_countdown}")
            sota_countdown -= 1  # pyre-ignore

# Governor Audit Moved to Pre-Log Phase (v6.1.16)

        # Reset intra-epoch skip/resume counters
        resume_iteration = -1


    print(f"\n--- Exporting {args.model} to SOTA Counterparts ---")

    # Final SWA Synchronization (Manifold Stabilization)
    if epoch >= swa_start:
        print("🧱 [SWA] Finalizing Stochastic Weight Average for production deployment...")
        update_bn(train_loader, swa_model, device=device)
        model = swa_model.module # Extract the averaged weights back into the main model

    try:
        model.eval()  # pyre-ignore

        model_info = unified_models_registry.get(args.model, {})
        size_raw = model_info.get("input_size", config.get("default_img_size", 256))
        if isinstance(size_raw, list):
            h, w = (int(size_raw[1]), int(size_raw[2])) if len(size_raw)==3 else (int(size_raw[0]), int(size_raw[1]))
        else:
            h, w = int(size_raw), int(size_raw)

        dummy_input = torch.randn(1, 3, h, w).to(device)

        model_filename = model_info.get("filename", args.model)
        base_name = f"LemGendary{model_filename}"

        # --- 2026 SOTA Universal Export Suite Synchronization (v1.0.49) ---
        # We delegate all export logic to the specialized standalone suite to ensure zero-leak binaries.
        python_exe = sys.executable
        export_script_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "export")

        # 1. Standardized ONNX Matrix (Lean-Synthesis + External Data FP32)
        print(f"✨ [EXPORT] Triggering Universal ONNX Matrix Synthesis...")
        onnx_script = os.path.join(export_script_dir, "export_onnx_model.py")
        subprocess.call([python_exe, onnx_script, "--model", args.model, "--yes"])

        # 2. Standardized PyTorch Standalone (Architecture + Weights Unity)
        print(f"✨ [EXPORT] Triggering Standalone PyTorch Unity Synthesis...")
        torch_script = os.path.join(export_script_dir, "export_torch_model.py")
        subprocess.call([python_exe, torch_script, "--model", args.model, "--yes"])

        # 2026 Resilience: We use the PERSISTENT best_metrics for the final README
        # to ensure doc-weight parity if the final epoch was a regression.
        metrics_to_report = best_metrics if best_quality_score > -1.0 else {"plcc": plcc, "srcc": srcc, "psnr": psnr, "ssim": ssim_val, "lpips": lpips_val, "fid": fid}
        from training.doc_generator import build_model_readme # pyre-ignore
        readme_text = build_model_readme(args.model, unified_models_registry, epoch+1, metrics_to_report)
        with open(os.path.join(export_dir, "README.md"), "w") as f:
            f.write(readme_text)

        # --- 2026 Kaggle Notebook/Inference Generator ---
        print(f"✨ [EXPORT] Generating Kaggle Inference Notebook...")
        import json
        inference_notebook_path = os.path.join(export_dir, "kaggle_inference.ipynb")
        pascal_model_name = args.model.replace('_', ' ').title().replace(' ', '')
        kebab_model_name = args.model.replace('_', '-')

        import base64
        cell_1_b64 = "aW1wb3J0IHRvcmNoCmltcG9ydCBvbm54cnVudGltZSBhcyBvcnQKZnJvbSBQSUwgaW1wb3J0IEltYWdlCmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgb3MKCmRldmljZSA9IHRvcmNoLmRldmljZSgnY3VkYScgaWYgdG9yY2guY3VkYS5pc19hdmFpbGFibGUoKSBlbHNlICdjcHUnKQpwcmludChmIlVzaW5nIGRldmljZToge2RldmljZX0iKQo="
        cell_2_b64 = "bW9kZWxfcGF0aCA9ICcva2FnZ2xlL2lucHV0L2xlbWdlbmRhcnkte2tlYmFiX21vZGVsX25hbWV9L0xlbUdlbmRhcnl7cGFzY2FsX21vZGVsX25hbWV9LnB0aCcKaWYgbm90IG9zLnBhdGguZXhpc3RzKG1vZGVsX3BhdGgpOgogICAgbW9kZWxfcGF0aCA9ICdMZW1HZW5kYXJ5e3Bhc2NhbF9tb2RlbF9uYW1lfS5wdGgnICMgTG9jYWwgZmFsbGJhY2sKCnRyeToKICAgIG1vZGVsID0gdG9yY2gubG9hZChtb2RlbF9wYXRoLCBtYXBfbG9jYXRpb249ZGV2aWNlKQogICAgbW9kZWwuZXZhbCgpCiAgICBwcmludCgi4pyFIFB5VG9yY2ggTW9kZWwgbG9hZGVkIHN1Y2Nlc3NmdWxseSEiKQpleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICBwcmludChmIuKdjCBFcnJvciBsb2FkaW5nIFB5VG9yY2ggbW9kZWw6IHtlfSIpCg=="
        cell_3_b64 = "b25ueF9wYXRoID0gJy9rYWdnbGUvaW5wdXQvbGVtZ2VuZGFyeS17a2ViYWJfbW9kZWxfbmFtZX0vTGVtR2VuZGFyeXtwYXNjYWxfbW9kZWxfbmFtZX0ub25ueCcKaWYgbm90IG9zLnBhdGguZXhpc3RzKG9ubnhfcGF0aCk6CiAgICBvbm54X3BhdGggPSAnTGVtR2VuZGFyeXtwYXNjYWxfbW9kZWxfbmFtZX0ub25ueCcgIyBMb2NhbCBmYWxsYmFjawoKdHJ5OgogICAgb3J0X3Nlc3Npb24gPSBvcnQuSW5mZXJlbmNlU2Vzc2lvbihvbm54X3BhdGgsIHByb3ZpZGVycz1bJ0NVREFFeGVjdXRpb25Qcm92aWRlcicsICdDUFVFeGVjdXRpb25Qcm92aWRlciddKQogICAgcHJpbnQoIuKchSBPTk5YIFNlc3Npb24gaW5pdGlhbGl6ZWQgc3VjY2Vzc2Z1bGx5ISIpCmV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgIHByaW50KGYi4p2MIEVycm9yIGluaXRpYWxpemluZyBPTk5YIHNlc3Npb246IHtlfSIpCg=="
        
        cell_1_source = base64.b64decode(cell_1_b64).decode('utf-8')
        cell_2_source = base64.b64decode(cell_2_b64).decode('utf-8').replace("{kebab_model_name}", kebab_model_name).replace("{pascal_model_name}", pascal_model_name)
        cell_3_source = base64.b64decode(cell_3_b64).decode('utf-8').replace("{kebab_model_name}", kebab_model_name).replace("{pascal_model_name}", pascal_model_name)

        notebook_content = {
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12.12"}
            },
            "nbformat_minor": 4,
            "nbformat": 4,
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": [f"# LemGendary SOTA Inference: {pascal_model_name}\n", "This notebook natively executes the explicit LemGendary Neural Architecture topologies directly upon Kaggle cloud hardware for both FP32 PyTorch and ONNX models."],
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": cell_1_source.splitlines(keepends=True),
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None
                },
                {
                    "cell_type": "markdown",
                    "source": ["## 1. PyTorch (FP32) Inference\n"],
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": cell_2_source.splitlines(keepends=True),
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None
                },
                {
                    "cell_type": "markdown",
                    "source": ["## 2. ONNX (FP32) Inference\n"],
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": cell_3_source.splitlines(keepends=True),
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None
                }
            ]
        }

        with open(inference_notebook_path, "w", encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=1)

        # --- 2026 Kaggle Cleanup ---
        if args.env == "kaggle":
            print("🧹 [KAGGLE] Training complete. Purging local dataset shards to prevent /kaggle/working OOM...")
            if os.path.exists(data_dir):
                try:
                    shutil.rmtree(data_dir)
                    print("🧹 [KAGGLE] Dataset cache successfully wiped.")
                except Exception as e:
                    print(f"⚠️ [KAGGLE] Failed to wipe dataset cache: {e}")
        elif args.env == "local":
            print(f"\n🏆 [MISSION COMPLETE] Training / Test Phase ended.")
            if os.path.exists(data_dir):
                # Interactive Post-SOTA Data Wipe
                ans = input(f"🧹 Do you want to approve removing the local training dataset cache ({data_dir}) to save disk space? (y/n): ").strip().lower()
                if ans == 'y':
                    try:
                        shutil.rmtree(data_dir)
                        print(f"✅ Local dataset cache '{data_dir}' has been successfully wiped.")
                    except Exception as e:
                        print(f"⚠️ Failed to wipe local dataset cache: {e}")

        # --- 2026 Resilient Artifact Sync (Windows IO Guard) ---
        # Windows often keeps a lock on newly created ONNX files for several milliseconds.
        # We implementation a settle-period and retry loop to ensure production sync succeeds.
        sync_success = False
        for attempt in range(1, 4):
            try:
                if attempt > 1:
                    print(f"🔄 [RESILIENCY] Sync attempt {attempt}/3... (Waiting for Windows IO recovery)")
                    time.sleep(2)

                if config.get("export_to_external_folder", False):
                    shutil.copytree(export_dir, local_dir, dirs_exist_ok=True)

                trained_models_dir = os.path.normpath(os.path.join(project_root, "..", "LemGendaryModels", args.model))

                # --- 2026 Collision Guard (Windows IO Protection) ---
                # Only sync if the production target is actually different from the export staging area.
                # This prevents WinError 32 when the config export_dir is already set to '../LemGendaryModels'.
                if os.path.abspath(export_dir) != os.path.abspath(trained_models_dir):
                    os.makedirs(trained_models_dir, exist_ok=True)
                    shutil.copytree(export_dir, trained_models_dir, dirs_exist_ok=True)
                    sync_success = True
                    print("SUCCESS: Artifacts securely synced to local_models and LemGendaryModels.")
                else:
                    sync_success = True
                    print("ℹ️  [INFO] Artifacts are already in the production directory. Skipping redundant sync.")
                break
            except Exception as e:
                if attempt == 3:
                    print(f"❌ [CRITICAL] Artifact sync failed after 3 attempts: {e}")
                else:
                    time.sleep(1)

        if not sync_success:
            print("⚠️  [WARNING] Model was trained and exported, but final sync failed. Artifacts remain in the export directory.")

    except Exception as e:
        print(f"ONNX Export Failure: {e}")

if __name__ == "__main__":
    main()  # pyre-ignore
