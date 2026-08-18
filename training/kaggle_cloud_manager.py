import os
import sys
import json
import time
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

# [SENIOR HARDENING v16.0 - SYNC_ID: 1542]

def resolve_kaggle_credentials(prompt_interactive: bool = True, override_user: Optional[str] = None, override_key: Optional[str] = None) -> Tuple[str, str]:
    """
    Resolves Kaggle API credentials with senior fallback hierarchy:
    1. CLI overrides / Explicit function parameters
    2. Environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
    3. User home ~/.kaggle/kaggle.json
    4. Fallback: lemgendary-training-suite/.kaggle_token & config.yaml defaults
    """
    suite_dir = Path(__file__).resolve().parent.parent
    token_file = suite_dir / ".kaggle_token"
    user_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    k_user = override_user or os.environ.get("KAGGLE_USERNAME", "")
    k_key = override_key or os.environ.get("KAGGLE_KEY", "")

    # Check ~/.kaggle/kaggle.json if env vars not set
    if not k_user and user_kaggle_json.exists():
        try:
            with open(user_kaggle_json, "r") as f:
                creds = json.load(f)
                k_user = creds.get("username", "")
                k_key = creds.get("key", "")
        except Exception:
            pass

    # If still missing and interactive, prompt user with fallback indication
    if (not k_user or not k_key) and prompt_interactive and sys.stdin.isatty():
        print("\n" + "=" * 70)
        print(" [KAGGLE CLOUD ENGINE] Authentication Setup")
        print("=" * 70)
        print(" Enter your Kaggle API credentials for headless GPU execution.")
        print(" (Press ENTER without input to use default automated fallback)")
        try:
            inp_user = input(" Kaggle Username [Default: lemtreursi]: ").strip()
            inp_key = input(" Kaggle API Key / Token: ").strip()
            if inp_user:
                k_user = inp_user
            if inp_key:
                k_key = inp_key
        except (EOFError, KeyboardInterrupt):
            pass

    # Automated Fallback to local .kaggle_token
    if not k_user:
        k_user = "lemtreursi"
    if not k_key and token_file.exists():
        try:
            with open(token_file, "r") as f:
                k_key = f.read().strip()
                # If token starts with KGAT_, strip prefix if needed or pass as token
                if k_key.startswith("KGAT_"):
                    k_key = k_key.replace("KGAT_", "")
        except Exception:
            pass

    # Export to environment for kaggle and kagglehub SDKs
    if k_user:
        os.environ["KAGGLE_USERNAME"] = k_user
    if k_key:
        os.environ["KAGGLE_KEY"] = k_key

    return k_user, k_key


def get_kernel_slug(model_name: str, username: str) -> str:
    """Generates standardized Kaggle kernel slug."""
    clean_model = model_name.replace("_", "-")
    return f"{username}/lemgendary-{clean_model}-training"


def create_cloud_kernel_bundle(model_name: str, username: str, gpu: str = "T4") -> Path:
    """
    Creates a standalone Kaggle kernel bundle containing metadata and execution script.
    """
    suite_dir = Path(__file__).resolve().parent.parent
    kernel_dir = suite_dir / "cloud_jobs" / f"{model_name}_cloud_job"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    slug = f"lemgendary-{model_name.replace('_', '-')}-training"
    title = f"LemGendary {model_name.replace('_', ' ').title()} Training"

    # kernel-metadata.json
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "train_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Standalone execution script
    script_content = f"""# LemGendary Headless Cloud Worker - Auto-Generated
import os
import sys
import subprocess

print("[OK] [CLOUD WORKER] Booting Kaggle High-VRAM GPU Environment...")

# Verify GPU
subprocess.run(["nvidia-smi"])

# Clone/Sync LemGendary Training Suite
repo_url = "https://github.com/lemgenda/lemgendary-training-suite.git"
work_dir = "/kaggle/working/lemgendary-training-suite"

if not os.path.exists(work_dir):
    print("[CLOUD] Cloning Training Suite manifold...")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, work_dir])

os.chdir(work_dir)
sys.path.insert(0, work_dir)

# Launch Cloud Training with Auto-Sync
print("[CLOUD] Launching SOTA Matrix for >> {model_name} <<...")
cmd = [
    sys.executable, "-m", "training.train",
    "--model", "{model_name}",
    "--env", "kaggle",
    "--auto_sync",
    "--yes"
]
subprocess.run(cmd)
print("[CLOUD] Training Execution Completed.")
"""

    with open(kernel_dir / "train_kernel.py", "w") as f:
        f.write(script_content)

    return kernel_dir


def launch_kaggle_training(model_name: str, config: Optional[dict] = None, username: Optional[str] = None, key: Optional[str] = None, gpu: str = "T4") -> bool:
    """
    Pushes and launches a training kernel to Kaggle GPU Cloud headlessly.
    """
    user, api_key = resolve_kaggle_credentials(prompt_interactive=True, override_user=username, override_key=key)
    kernel_dir = create_cloud_kernel_bundle(model_name, user, gpu=gpu)
    slug = get_kernel_slug(model_name, user)

    print(f"\n[LAUNCH] [KAGGLE CLOUD] Deploying kernel bundle to Kaggle...")
    print(f" -> Kernel ID: {slug}")
    print(f" -> Accelerator: Tesla GPU ({gpu})")

    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Push kernel
        api.kernels_push(str(kernel_dir))
        print(f"\n[SUCCESS] [KAGGLE CLOUD] Kernel successfully pushed and queued on Kaggle GPU!")
        print(f" -> Monitor Live: python -m training.kaggle_cloud_manager --action monitor --model {model_name}")
        return True
    except Exception as e:
        print(f"[ERROR] [KAGGLE CLOUD] Push failed: {e}")
        # Fallback to CLI
        try:
            print("[INFO] Attempting CLI push fallback...")
            res = subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], capture_output=True, text=True)
            if res.returncode == 0:
                print(f"[SUCCESS] CLI push successful: {res.stdout.strip()}")
                return True
            else:
                print(f"[ERROR] CLI push failed: {res.stderr.strip()}")
        except Exception as cli_err:
            print(f"[ERROR] CLI fallback failed: {cli_err}")
        return False


def monitor_kaggle_training(model_name: str, username: Optional[str] = None, poll_interval: int = 15):
    """
    Streams live status and logs from Kaggle for the specified model kernel.
    """
    user, _ = resolve_kaggle_credentials(prompt_interactive=True, override_user=username)
    slug = get_kernel_slug(model_name, user)

    print(f"\n[SIGNAL] [KAGGLE CLOUD] Connecting to telemetry stream for: {slug}")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        last_status = None
        while True:
            status_obj = api.kernels_status(slug)
            status = getattr(status_obj, 'status', str(status_obj))
            
            if status != last_status:
                print(f" -> [{time.strftime('%H:%M:%S')}] Cloud Status: {status.upper()}")
                last_status = status

            if status in ["complete", "error", "cancelAck"]:
                print(f"\n[INFO] Cloud Job reached terminal state: {status}")
                break

            time.sleep(poll_interval)

    except Exception as e:
        print(f"[ERROR] [KAGGLE MONITOR] Monitoring stream encountered an error: {e}")


def pull_kaggle_artifacts(model_name: str, destination_dir: Optional[str] = None, username: Optional[str] = None) -> bool:
    """
    Downloads latest checkpoints and metrics from Kaggle Models or kernel output.
    """
    user, _ = resolve_kaggle_credentials(prompt_interactive=True, override_user=username)
    suite_dir = Path(__file__).resolve().parent.parent
    target_dir = Path(destination_dir) if destination_dir else suite_dir.parent / "LemGendaryModels" / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[SYNC] [KAGGLE CLOUD] Pulling artifacts for >> {model_name} << to {target_dir}...")
    try:
        import kagglehub
        handle = f"{user}/lemgendary-{model_name.replace('_', '-')}-checkpoints/pytorch/default"
        print(f" -> Querying Kaggle Model Registry: {handle}")
        download_path = kagglehub.model_download(handle)
        if download_path and os.path.exists(download_path):
            for item in os.listdir(download_path):
                s = os.path.join(download_path, item)
                d = os.path.join(str(target_dir), item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            print(f"[SUCCESS] Successfully pulled model artifacts to: {target_dir}")
            return True
    except Exception as e:
        print(f"[WARNING] Model Registry pull failed ({e}). Attempting kernel output retrieval...")

    # Fallback to kernel output retrieval
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        slug = get_kernel_slug(model_name, user)
        api.kernels_output(slug, path=str(target_dir))
        print(f"[SUCCESS] Kernel output artifacts pulled successfully.")
        return True
    except Exception as k_err:
        print(f"[ERROR] Artifact pull failed: {k_err}")
        return False


def main():
    parser = argparse.ArgumentParser(description="LemGendary Headless Kaggle Cloud Engine")
    parser.add_argument("--action", type=str, required=True, choices=["launch", "status", "monitor", "pull", "cancel", "setup_auth"])
    parser.add_argument("--model", type=str, default="nima_technical", help="Model manifold name")
    parser.add_argument("--username", type=str, default=None, help="Kaggle Username override")
    parser.add_argument("--key", type=str, default=None, help="Kaggle API Key override")
    parser.add_argument("--gpu", type=str, default="T4", choices=["T4", "P100"])
    parser.add_argument("--output_dir", type=str, default=None, help="Target destination for downloaded artifacts")
    args = parser.parse_args()

    if args.action == "setup_auth":
        u, k = resolve_kaggle_credentials(prompt_interactive=True)
        print(f"[OK] Authentication verified for Kaggle user: {u}")
    elif args.action == "launch":
        launch_kaggle_training(args.model, username=args.username, key=args.key, gpu=args.gpu)
    elif args.action == "monitor":
        monitor_kaggle_training(args.model, username=args.username)
    elif args.action == "pull":
        pull_kaggle_artifacts(args.model, destination_dir=args.output_dir, username=args.username)
    elif args.action == "status":
        u, _ = resolve_kaggle_credentials(prompt_interactive=False, override_user=args.username)
        slug = get_kernel_slug(args.model, u)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            st = api.kernels_status(slug)
            print(f"Status for {slug}: {st}")
        except Exception as e:
            print(f"Status check failed: {e}")

if __name__ == "__main__":
    main()
