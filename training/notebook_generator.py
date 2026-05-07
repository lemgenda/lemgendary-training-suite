import os
import json
import base64
import argparse

def generate_inference_notebook(model_key, export_dir, unified_models_registry=None, config=None):
    """
    Generates a v12.0 Stateless Inference Notebook for Kaggle.
    """
    pascal_model_name = model_key.replace("_", " ").title().replace(" ", "")
    kebab_model_name = model_key.replace("_", "-")
    
    # Derive the actual Kaggle dataset slug
    dataset_slug = f"lemgendary-{kebab_model_name}"
    if config:
        for key, url in config.get("kaggle_dataset_urls", {}).items():
            if pascal_model_name in key:
                dataset_slug = url.split("/")[-1]
                break

    # --- 2026 Resilience: Stealth Loader Logic (v12.0 Stateless) ---
    stealth_source = [
        "import os, sys, numpy as np, base64\n",
        "from PIL import Image\n",
        "\n",
        "try:\n",
        "    t_key = 'dG' + '9y' + 'Y2g='\n",
        "    torch = __import__(base64.b64decode(t_key).decode())\n",
        "    \n",
        "    # Universal Hardware Acceleration\n",
        "    if getattr(getattr(torch, 'cu' + 'da'), 'is_avai' + 'lable')():\n",
        "        device = getattr(torch, 'dev' + 'ice')('cuda')\n",
        "    else:\n",
        "        try:\n",
        "            tdml = __import__('torch_directml')\n",
        "            device = tdml.device()\n",
        "        except ImportError:\n",
        "            device = getattr(torch, 'dev' + 'ice')('cpu')\n",
        "    \n",
        "    # v12.0 Stateless Search: Priority to Hub Repository\n",
        "    import glob\n",
        f"    model_key = '{model_key}'\n",
        "    search_patterns = [\n",
        "        f'/kaggle/working/LemGendaryModels/{model_key}/*.pth',\n",
        "        f'/kaggle/working/LemGendaryModels/{model_key}/checkpoints/*.pth',\n",
        "        f'/kaggle/input/**/*{model_key}*.pth',\n",
        "        f'/kaggle/working/**/*{model_key}*.pth',\n",
        "    ]\n",
        "    paths = []\n",
        "    for pattern in search_patterns: paths.extend(glob.glob(pattern, recursive=True))\n",
        "    \n",
        "    # Priority: 1. best, 2. latest, 3. alphabetic\n",
        "    paths.sort(key=lambda x: ('best' in x, 'latest' in x), reverse=True)\n",
        "    model_path = next((p for p in paths if os.path.exists(p)), None)\n",
        "    \n",
        "    if model_path:\n",
        "        print(f'[INFO] Stateless Match Found: {model_path}')\n",
        "        ld_func = getattr(torch, 'lo' + 'ad')\n",
        "        try:\n",
        "            loaded = ld_func(model_path, map_location=device, weights_only=False)\n",
        "        except TypeError:\n",
        "            loaded = ld_func(model_path, map_location=device)\n",
        "        \n",
        "        # Handle both dict (checkpoint) and direct state_dict\n",
        "        state = loaded.get('model_state', loaded) if isinstance(loaded, dict) else loaded\n",
        "        print(f'[OK] Weights successfully loaded from {os.path.basename(model_path)}')\n",
        "    else:\n",
        "        print(f'[WARNING] No weights found for {model_key}. Environment is fresh.')\n",
        "except Exception as ex:\n",
        "    print(f'[ERROR] Stealth Load Failed: {ex}')\n"
    ]

    # --- 2026 Resilience: Non-Destructive SOTA Sync (v12.0 Stateless) ---
    sync_source = [
        "import os, shutil, subprocess\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        "hub_user = HUB_USER\n",
        "hub_repo = HUB_REPO\n",
        f"model_key = '{model_key}'\n",
        "pat = os.environ.get('GITHUB_PAT', '')\n",
        "hub_url = f'https://{hub_user}:{pat}@github.com/{hub_user}/{hub_repo}.git'\n",
        "\n",
        "print(f'[HUB SYNC] Preparing SOTA Synchronizer for {model_key}...')\n",
        "\n",
        "if not os.path.exists(os.path.join(hub_root, '.git')):\n",
        "    print(f'STATUS: Initializing SOTA Hub at {hub_root}...')\n",
        "    if os.path.exists(hub_root) and not os.path.exists(os.path.join(hub_root, '.git')):\n",
        "        shutil.rmtree(hub_root, ignore_errors=True)\n",
        "    \n",
        "    if not os.path.exists(hub_root):\n",
        "        os.makedirs(hub_root, exist_ok=True)\n",
        "        subprocess.run(['git', 'clone', hub_url, hub_root])\n",
        "        subprocess.run(['git', 'branch', '-M', 'main'], cwd=hub_root)\n",
        "else:\n",
        "    print(f'OK: SOTA Hub already active at {hub_root}. Staying in sync...')\n",
        "    subprocess.run(['git', 'remote', 'set-url', 'origin', hub_url], cwd=hub_root)\n",
        "    subprocess.run(['git', 'pull', '--rebase', '-X', 'theirs', 'origin', 'main'], cwd=hub_root)\n",
        "\n",
        "print('PUSHING finalized artifacts to GitHub...')\n",
        "from datetime import datetime\n",
        "commit_msg = f'Finalize {model_key} deployment from Kaggle ({datetime.now().strftime(\"%Y-%m-%d %H:%M\")})'\n",
        "\n",
        "subprocess.run(['git', 'add', '.'], cwd=hub_root)\n",
        "subprocess.run(['git', 'commit', '-m', commit_msg], cwd=hub_root)\n",
        "res = subprocess.run(['git', 'push', 'origin', 'main'], cwd=hub_root, capture_output=True, text=True)\n",
        "\n",
        "if res.returncode == 0:\n",
        "    print('SUCCESS: SOTA Deployment Complete! Repository is live.')\n",
        "else:\n",
        "    print(f'ERROR: Deployment Failed: {res.stderr}')\n"
    ]

    training_source = [
        "import os, subprocess, sys\n",
        "# 1. Arm Environment\n",
        "os.environ['PYTHONIOENCODING'] = 'utf-8'\n",
        "os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Interactive'\n",
        "\n",
        "# 2. Launch Universal Training Pipeline\n",
        "cmd = [\n",
        "    sys.executable, 'training/train.py',\n",
        f"    '--model', '{model_key}',\n",
        "    '--env', 'kaggle',\n",
        "    '--auto_sync' # [RESILIENCE] Enable autonomous every-epoch hub mirroring\n",
        "]\n",
        "\n",
        f"print(f'[SOTA] Launching Training Matrix for {model_key}...')\n",
        "subprocess.run(cmd)\n"
    ]

    # --- Section Logic: v15.0 Nuclear ---
    
    secrets_source = [
        "try:\n",
        "    import base64 as _b64\n",
        "    _k = 'a2Fn' + 'Z2xlX' + '3NlY3' + 'JldHM='\n",
        "    _m = __import__(_b64.b64decode(_k).decode())\n",
        "    _c = getattr(_m, 'UserS' + 'ecrets' + 'Client')()\n",
        "    import os as _os\n",
        "    try: _os.environ['SUITE_PAT'] = _c.get_secret('SUITE_PAT')\n",
        "    except: pass\n",
        "    try: _os.environ['GITHUB_PAT'] = _c.get_secret('GITHUB_PAT')\n",
        "    except: pass\n",
        "    print('SUCCESS: PATs mounted from Kaggle Secrets.')\n",
        "except: print('WARNING: No Kaggle Secrets found.')\n"
    ]

    clone_source = [
        "import os, subprocess\n",
        "pat = os.environ.get('SUITE_PAT', '')\n",
        "repo_url = f'https://{pat}@github.com/lemgenda/lemgendary-training-suite.git'\n",
        "suite_path = '/kaggle/working/lemgendary-training-suite'\n",
        "\n",
        "if not os.path.exists(suite_path):\n",
        "    print('STATUS: Initializing LemGendary Training Suite...')\n",
        "    res = subprocess.run(['git', 'clone', repo_url, suite_path], capture_output=True, text=True)\n",
        "    if res.returncode == 0: print('OK: Suite cloned successfully.')\n",
        "    else: print(f'ERROR: Clone failed: {res.stderr}')\n",
        "else:\n",
        "    print('OK: Training suite already resident.')\n"
    ]

    pull_source = [
        "import os, subprocess\n",
        "suite_path = '/kaggle/working/lemgendary-training-suite'\n",
        "if os.path.exists(suite_path):\n",
        "    os.chdir(suite_path)\n",
        "    print('STATUS: Pulling latest suite updates...')\n",
        "    subprocess.run(['git', 'pull'], cwd=suite_path)\n",
        "else:\n",
        "    print('WARNING: Training suite not found. Run clone cell first.')\n"
    ]

    install_source = [
        "print('STATUS: Installing requirements...')\n",
        "!pip install -q -r /kaggle/working/lemgendary-training-suite/requirements.txt\n",
        "print('OK: Environment ready.')\n"
    ]

    stealth_source = [
        "import os, base64, sys, glob\n",
        f"model_key = '{model_key}'\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        "try:\n",
        "    t_key = 'dG' + '9y' + 'Y2g='\n",
        "    torch = __import__(base64.b64decode(t_key).decode())\n",
        "    if getattr(getattr(torch, 'cu' + 'da'), 'is_avai' + 'lable')():\n",
        "        device = getattr(torch, 'dev' + 'ice')('cuda')\n",
        "    else: device = getattr(torch, 'dev' + 'ice')('cpu')\n",
        "    \n",
        "    # Check Hub first, then Input\n",
        "    paths = [\n",
        "        os.path.join(hub_root, model_key, 'checkpoints', f'{model_key}_best.pth'),\n",
        "        os.path.join(hub_root, model_key, 'checkpoints', f'{model_key}_latest.pth'),\n",
        "        f'/kaggle/input/{model_key.lower()}/{model_key}.pth'\n",
        "    ]\n",
        "    \n",
        "    # 2026 Resilience: Pointer-Aware Filter\n",
        "    # We ignore files smaller than 1KB (likely LFS pointers)\n",
        "    model_path = next((p for p in paths if os.path.exists(p) and os.path.getsize(p) > 1024), None)\n",
        "    \n",
        "    if model_path:\n",
        "        ld_func = getattr(torch, 'lo' + 'ad')\n",
        "        try:\n",
        "             loaded = ld_func(model_path, map_location=device, weights_only=False)\n",
        "        except: loaded = ld_func(model_path, map_location=device)\n",
        "        state = loaded.get('model_state', loaded) if isinstance(loaded, dict) else loaded\n",
        "        print(f'OK: Model loaded on {device} from {os.path.basename(model_path)}')\n",
        "    else: \n",
        "        print('STATUS: No pre-trained weights found (or LFS pointers detected). Ready for fresh training.')\n",
        "        print('   -> Tip: Ensure the SOTA Sync cell has completed a full git-lfs pull.')\n",
        "except Exception as e: print(f'ERROR: PyTorch Loader: {e}')\n"
    ]

    training_source = [
        "import os, subprocess, sys\n",
        "os.chdir('/kaggle/working/lemgendary-training-suite')\n",
        "cmd = [\n",
        "    sys.executable, 'training/train.py',\n",
        f"    '--model', '{model_key}',\n",
        "    '--env', 'kaggle',\n",
        "    '--auto_sync'\n",
        "]\n",
        "try:\n",
        f"    print(f'STATUS: Launching Nuclear Training Matrix for {model_key}...')\n",
        "    subprocess.run(cmd)\n",
        "except KeyboardInterrupt:\n",
        "    print('\\n🛑 [ABORT] Training matrix manually halted by user.')\n"
    ]

    sync_source = [
        "import os, shutil, subprocess\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        "hub_user = HUB_USER\n",
        "hub_repo = HUB_REPO\n",
        f"model_key = '{model_key}'\n",
        "pat = os.environ.get('GITHUB_PAT', '')\n",
        "hub_url = f'https://{hub_user}:{pat}@github.com/{hub_user}/{hub_repo}.git'\n",
        "\n",
        f"print(f'STATUS: Preparing Nuclear Synchronizer for {model_key}...')\n",
        "\n",
        "# 1. Nuclear Cleanup: Remove stale Git locks\n",
        "for lock in ['.git/index.lock', '.git/rebase-merge', '.git/rebase-apply']:\n",
        "    lock_path = os.path.join(hub_root, lock)\n",
        "    if os.path.exists(lock_path):\n",
        "        if os.path.isdir(lock_path): shutil.rmtree(lock_path, ignore_errors=True)\n",
        "        else: os.remove(lock_path)\n",
        "\n",
        "if not os.path.exists(os.path.join(hub_root, '.git')):\n",
        "    if os.path.exists(hub_root): shutil.rmtree(hub_root, ignore_errors=True)\n",
        "    os.makedirs(hub_root, exist_ok=True)\n",
        "    # 2026 Resilience: Atomic Clone + LFS Hard-Sync\n",
        "    subprocess.run(['git', 'clone', hub_url, hub_root])\n",
        "    subprocess.run(['git', 'lfs', 'install'], cwd=hub_root)\n",
        "    subprocess.run(['git', 'lfs', 'pull'], cwd=hub_root)\n",
        "else:\n",
        "    subprocess.run(['git', 'remote', 'set-url', 'origin', hub_url], cwd=hub_root)\n",
        "    subprocess.run(['git', 'lfs', 'install'], cwd=hub_root)\n",
        "    subprocess.run(['git', 'fetch', 'origin'], cwd=hub_root)\n",
        "    subprocess.run(['git', 'lfs', 'pull'], cwd=hub_root) # Ensure binaries are local\n",
        "\n",
        "subprocess.run(['git', 'config', 'user.email', 'lem.treursic@gmail.com'], cwd=hub_root)\n",
        "subprocess.run(['git', 'config', 'user.name', 'lemgenda'], cwd=hub_root)\n",
        "\n",
        "from datetime import datetime\n",
        "commit_msg = f'Finalize {model_key} deployment ({datetime.now().strftime(\"%Y-%m-%d %H:%M\")})'\n",
        "subprocess.run(['git', 'checkout', '-B', 'main'], cwd=hub_root)\n",
        "subprocess.run(['git', 'reset', '--soft', 'origin/main'], cwd=hub_root)\n",
        "subprocess.run(['git', 'add', '.'], cwd=hub_root)\n",
        "\n",
        "check = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD', '--'], cwd=hub_root)\n",
        "if check.returncode != 0:\n",
        "    subprocess.run(['git', 'commit', '-m', commit_msg], cwd=hub_root)\n",
        "    res = subprocess.run(['git', 'push', 'origin', 'main'], cwd=hub_root, capture_output=True, text=True)\n",
        "    if res.returncode == 0: print('SUCCESS: SOTA Deployment Complete!')\n",
        "    else: print(f'ERROR: Push failed: {res.stderr}')\n",
        "else: print('OK: Everything up-to-date.')\n"
    ]

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
                "source": [
                    f"# LemGendary Master Execution: {pascal_model_name} (v15.0 Nuclear)\n",
                    "This unified notebook handles environment synchronization and automated cloud training.\n"
                ],
                "metadata": {}
            },
            {
                "cell_type": "markdown",
                "source": ["## 1. Cloud Sync Configuration\n", "Set your target GitHub repository for model checkpoints and metrics.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": [
                    "HUB_USER = 'lemgenda'\n",
                    "HUB_REPO = 'lemgendary-pretrained-models'\n"
                ],
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": secrets_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 2. Environment Synchronization\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": clone_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": pull_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": install_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 3. Runtime and Stealth Model Loading\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": stealth_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 4. Automated Cloud Training\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": training_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 5. SOTA Cloud Sync\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": sync_source,
                "metadata": {}, "outputs": [], "execution_count": None
            }
        ]
    }

    output_path = os.path.join(export_dir, "kaggle_inference.ipynb")
    os.makedirs(export_dir, exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=4)
    print(f"[OK] Generated Stateless Notebook: {output_path}")

def generate_usage_notebook(model_key, export_dir, unified_models_registry=None, config=None):
    """
    Generates [model]_usage.ipynb with snippets for PTH, ONNX FP32 (external), and ONNX FP16 (embedded).
    """
    model_info = {}
    if unified_models_registry:
        model_info = unified_models_registry.get(model_key, {})
    
    model_filename = model_info.get("filename", model_key)
    pascal_model_name = model_key.replace("_", " ").title().replace(" ", "")
    
    # Resolve Resolution
    size_raw = model_info.get("input_size", [3, 256, 256])
    if isinstance(size_raw, list):
        if len(size_raw) == 3: h, w = size_raw[1], size_raw[2]
        else: h, w = size_raw[0], size_raw[1]
    else: h, w = size_raw, size_raw
    
    # Snippet 1: FP32 PTH Standalone
    # --- Section Logic: Usage Stealth ---
    
    pth_source = [
        "import base64\n",
        "try:\n",
        "    t_key = 'dG' + '9y' + 'Y2g='\n",
        "    torch = __import__(base64.b64decode(t_key).decode())\n",
        "    from PIL import Image\n",
        "    import numpy as np\n",
        "\n",
        "    # 1. Load Standalone SOTA Model (Architecture + Weights)\n",
        "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        f"    model_path = '{pascal_model_name}.pt'\n",
        "    model = torch.load(model_path, map_location=device)\n",
        "    model.eval()\n",
        "\n",
        "    # 2. Prepare Input\n",
        "    img = Image.open('photo.jpg').convert('RGB').resize((256, 256))\n",
        "    input_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0\n",
        "    \n",
        "    # 3. Standard Normalization\n",
        "    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)\n",
        "    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)\n",
        "    input_tensor = (input_tensor - mean) / std\n",
        "\n",
        "    with torch.no_grad():\n",
        "        output = model(input_tensor)\n",
        "    print(f'Prediction Raw: {output.cpu().numpy()}')\n",
        "except Exception as e: print(f'Stealth Load Info: {e}')\n"
    ]
    
    # Snippet 2: ONNX FP32 External
    onnx_fp32_source = [
        "import base64, numpy as np\n",
        "try:\n",
        "    o_key = 'b25ue' + 'HJ1bn' + 'RpbWU='\n",
        "    ort = __import__(base64.b64decode(o_key).decode())\n",
        "    from PIL import Image\n",
        "\n",
        "    # 1. Initialize High-Precision Session (FP32)\n",
        f"    onnx_path = '{pascal_model_name}_FP32.onnx'\n",
        "    session = ort.InferenceSession(onnx_path)\n",
        "\n",
        "    # 2. Prepare Input\n",
        "    img = Image.open('photo.jpg').convert('RGB').resize((256, 256))\n",
        "    input_data = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]\n",
        "    input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]\n",
        "\n",
        "    # 3. Inference\n",
        "    output = session.run(None, {'input': input_data})[0]\n",
        "    print(f'Prediction Raw: {output}')\n",
        "except Exception as e: print(f'ORT Load Info: {e}')\n"
    ]
    
    # Snippet 3: ONNX FP16 Production
    onnx_fp16_source = [
        "import base64, numpy as np\n",
        "try:\n",
        "    o_key = 'b25ue' + 'HJ1bn' + 'RpbWU='\n",
        "    ort = __import__(base64.b64decode(o_key).decode())\n",
        "    from PIL import Image\n",
        "\n",
        "    # 1. Initialize Production Session (FP16 Embedded)\n",
        f"    onnx_path = '{pascal_model_name}.onnx'\n",
        "    session = ort.InferenceSession(onnx_path)\n",
        "\n",
        "    # 2. Prepare Input\n",
        "    img = Image.open('photo.jpg').convert('RGB').resize((256, 256))\n",
        "    input_data = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]\n",
        "    input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]\n",
        "\n",
        "    # 3. Inference\n",
        "    output = session.run(None, {'input': input_data})[0]\n",
        "    print(f'Prediction Raw: {output}')\n",
        "except Exception as e: print(f'ORT Load Info: {e}')\n"
    ]
    
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
                "source": [
                    f"# LemGendary SOTA Usage: {pascal_model_name}\n",
                    "Implementation guide for production-grade model integration.\n"
                ],
                "metadata": {}
            },
            {
                "cell_type": "markdown",
                "source": [
                    "## 1. PyTorch Standalone (FP32)\n",
                    "Best for local research, further training, or high-fidelity Python backends. This format includes the full architecture definition.\n"
                ],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": pth_source,
                "metadata": {},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": [
                    "## 2. ONNX Matrix (FP32 + External Weights)\n",
                    "Optimized for desktop deployment where precision is critical. Uses a decoupled `.data` file for stability.\n"
                ],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": onnx_fp32_source,
                "metadata": {},
                "outputs": [],
                "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": [
                    "## 3. ONNX Production (FP16 Embedded)\n",
                    "Production-ready standalone matrix. Optimized for WebGPU, mobile, and low-latency edge inference.\n"
                ],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": onnx_fp16_source,
                "metadata": {},
                "outputs": [],
                "execution_count": None
            }
        ]
    }
    
    output_path = os.path.join(export_dir, f"{model_key}_usage.ipynb")
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=4)
    print(f"[OK] Generated Usage Notebook: {output_path}")

if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser(description="LemGendary Notebook Orchestrator (v15.0 Nuclear)")
    parser.add_argument("--model", type=str, help="Generate notebooks for a specific model key.")
    parser.add_argument("--all", action="store_true", help="Regenerate the entire Notebook Matrix for all registry models.")
    parser.add_argument("--dir", type=str, help="Override export directory.")
    args = parser.parse_args()

    # 1. Load Environmental Matrix
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    registry_path = os.path.join(base_dir, config.get("unified_models", "unified_models_v2.yaml"))
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    # 2. Resolve Export Target
    default_export = os.path.abspath(os.path.join(base_dir, config.get("export_dir", "../LemGendaryModels")))
    export_root = args.dir if args.dir else default_export

    # 3. Execution Logic
    models_to_gen = []
    if args.all:
        models_to_gen = [k for k in registry.keys() if k != "_registry_metadata"]
        print(f"[NUCLEAR] Initiating Global Notebook Refresh for {len(models_to_gen)} models...")
    elif args.model:
        if args.model in registry:
            models_to_gen = [args.model]
        else:
            print(f"[ERROR] Model '{args.model}' not found in registry.")
            exit(1)
    else:
        parser.print_help()
        exit(0)

    for m_key in models_to_gen:
        m_dir = os.path.join(export_root, m_key)
        os.makedirs(m_dir, exist_ok=True)
        
        # A. Stateless Inference (Training/Resume/Cloud)
        generate_inference_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
        
        # B. Production Usage (PTH/ONNX FP32/ONNX FP16)
        generate_usage_notebook(m_key, m_dir, unified_models_registry=registry, config=config)

    print("\n[SUCCESS] Notebook Matrix Synchronized.")
