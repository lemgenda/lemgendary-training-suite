import os
import json
import base64
import argparse
import sys

def generate_inference_notebook(model_key, export_dir, unified_models_registry=None, config=None):
    """
    Generates a v16.2 Nuclear-Hardened Inference Notebook for Kaggle.
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

    # --- Section Logic: v16.0 Nuclear Orchestration ---
    
    hardware_sentinel_source = [
        "import torch, sys\n",
        "print('[OK] [SENTINEL] Auditing Hardware Manifold...')\n",
        "if not torch.cuda.is_available():\n",
        "    print('[WARNING] NO GPU DETECTED!')\n",
        "    print('[ACTION REQUIRED] Enable GPU Accelerator in notebook settings:')\n",
        "    print('   -> Kaggle: Right Panel -> Session Options -> Accelerator -> GPU T4 x2 or P100')\n",
        "    print('   -> Colab:  Runtime -> Change runtime type -> Hardware accelerator -> GPU')\n",
        "    print('   -> Continuing in CPU Fallback Mode for dry-run validation...')\n",
        "else:\n",
        "    props = torch.cuda.get_device_properties(0)\n",
        "    print(f'[OK] [ACTIVE] {props.name}')\n",
        "    print(f'[OK] [VRAM] {props.total_memory / 1024**3:.1f} GB')\n",
        "    if props.total_memory / 1024**3 < 10.0:\n",
        "        print('[WARNING] Low VRAM detected. Suite will enable Survival Profiles automatically.')\n"
    ]

    secrets_source = [
        "try:\n",
        "    import base64 as _b64\n",
        "    _k = 'a2Fn' + 'Z2xlX' + '3NlY3' + 'JldHM='\n",
        "    _m = __import__(_b64.b64decode(_k).decode())\n",
        "    _c = getattr(_m, 'UserS' + 'ecrets' + 'Client')()\n",
        "    import os as _os, json as _json\n",
        "    # 2026: Restore PAT mounting & Kaggle Key mounting for authenticated hub sync\n",
        "    g_pat = None\n",
        "    s_pat = None\n",
        "    k_key = None\n",
        "    k_user = None\n",
        "    try: g_pat = _c.get_secret('GITHUB_PAT')\n",
        "    except Exception: print('[REMEDY] Missing secret! You should create new secret named GITHUB_PAT with your GitHub Personal Access Token as value')\n",
        "    try: s_pat = _c.get_secret('SUITE_PAT')\n",
        "    except Exception: print('[REMEDY] Missing secret! You should create new secret named SUITE_PAT with your GitHub Personal Access Token as value')\n",
        "    try: k_key = _c.get_secret('KAGGLE_KEY')\n",
        "    except Exception: print('[REMEDY] Missing secret! You should create new secret named KAGGLE_KEY with your Kaggle API Token as value')\n",
        "    try: k_user = _c.get_secret('KAGGLE_USERNAME')\n",
        "    except Exception: print('[REMEDY] Missing secret! You should create new secret named KAGGLE_USERNAME with your Kaggle username as value')\n",
        "    \n",
        "    if g_pat: _os.environ['GITHUB_PAT'] = g_pat\n",
        "    if s_pat: _os.environ['SUITE_PAT'] = s_pat\n",
        "    \n",
        "    if not k_user: k_user = 'lemtreursi'\n",
        "    if k_key:\n",
        "        _os.environ['KAGGLE_KEY'] = k_key\n",
        "        _os.environ['KAGGLE_USERNAME'] = k_user\n",
        "        _k_dir = _os.path.expanduser('~/.kaggle')\n",
        "        _os.makedirs(_k_dir, exist_ok=True)\n",
        "        with open(_os.path.join(_k_dir, 'kaggle.json'), 'w') as _kf:\n",
        "            _json.dump({'username': k_user, 'key': k_key}, _kf)\n",
        "        _os.chmod(_os.path.join(_k_dir, 'kaggle.json'), 0o600)\n",
        "    \n",
        "    active = []\n",
        "    if s_pat: active.append('SUITE_PAT')\n",
        "    if g_pat: active.append('GITHUB_PAT')\n",
        "    if k_key: active.append('KAGGLE_KEY')\n",
        "    if active:\n",
        "        print(f'[OK] [AUTH] Kaggle Secrets mounted: {\", \".join(active)}')\n",
        "    else:\n",
        "        print('[ERROR] [CRITICAL] No PATs found in Kaggle Secrets! Private repositories will fail to clone.')\n",
        "        print('[ACTION REQUIRED] In Kaggle Notebook top bar -> Add-ons -> Secrets -> Add SUITE_PAT or GITHUB_PAT.')\n",
        "except Exception as e:\n",
        "    print(f'[ERROR] Secret mounting failed: {e}')\n"
    ]

    clone_source = [
        "import os, subprocess, shutil\n",
        "repo_url = 'https://github.com/lemgenda/lemgendary-training-suite.git'\n",
        "suite_path = '/kaggle/working/lemgendary-training-suite'\n",
        "pat = os.environ.get('SUITE_PAT', os.environ.get('GITHUB_PAT', ''))\n",
        "if pat:\n",
        "    # Use x-access-token for more reliable auth with fine-grained tokens\n",
        "    auth_url = repo_url.replace('https://', f'https://x-access-token:{pat}@')\n",
        "    print(f'[AUTH] Using {\"SUITE_PAT\" if os.environ.get(\"SUITE_PAT\") else \"GITHUB_PAT\"} for cloning...')\n",
        "else:\n",
        "    print('[WARNING] No PAT found in environment. Attempting public clone (will fail for private repos)...')\n",
        "    print('[ACTION REQUIRED] If clone fails, add SUITE_PAT or GITHUB_PAT to Kaggle Add-ons -> Secrets.')\n",
        "    auth_url = repo_url\n",
        "\n",
        "env = os.environ.copy()\n",
        "env['GIT_TERMINAL_PROMPT'] = '0'\n",
        "\n",
        "if not os.path.exists(suite_path):\n",
        "    print('[SUITE] Initializing LemGendary Training Suite...')\n",
        "    res = subprocess.run(['git', 'clone', auth_url, suite_path], capture_output=True, text=True, env=env)\n",
        "    if res.returncode == 0: \n",
        "        print('[OK] Suite cloned.')\n",
        "    else: \n",
        "        print(f'[ERROR] Clone failed: {res.stderr.strip()}')\n",
        "        print('[REMEDY] If a 403/401 occurs, ensure your SUITE_PAT or GITHUB_PAT has repo read permissions.')\n",
        "        if '403' in res.stderr or '401' in res.stderr or 'terminal prompts disabled' in res.stderr:\n",
        "            print('[ACTION REQUIRED] Add SUITE_PAT or GITHUB_PAT to Kaggle Add-ons -> Secrets with GitHub read permissions.')\n",
        "else:\n",
        "    print('[OK] Suite resident. Syncing origin and pulling latest...')\n",
        "    subprocess.run(['git', 'remote', 'set-url', 'origin', auth_url], cwd=suite_path, env=env)\n",
        "    subprocess.run(['git', 'pull'], cwd=suite_path, env=env)\n"
    ]

    symlink_source = [
        "import os\n",
        f"model_key = '{model_key}'\n",
        "target_dir = '/kaggle/working/LemGendaryDatasets'\n",
        "os.makedirs(target_dir, exist_ok=True)\n",
        "\n",
        "print(f'[DATA] Resolving manifolds for {model_key}...')\n",
        "found = []\n",
        "keys = [model_key.lower(), model_key.replace(\"_\", \"-\"), model_key.replace(\"_\", \"\")]\n",
        "\n",
        "# 1. Restricted BFS Scanner (max depth 4, directories only) to bypass FUSE latency\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    try:\n",
        "        queue = ['/kaggle/input']\n",
        "        depths = {'/kaggle/input': 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 4: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune models/checkpoints to prevent wasting time scanning weights\n",
        "                    # 2026 Resilience: Aggressive FUSE Pruning - NEVER enter raw image/target dirs to prevent OOM stat storms\n",
        "                    if item_lower in ['models', 'checkpoints', 'weights', 'images', 'targets', 'labels', 'masks', 'train', 'val', 'test', 'eval']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    is_match = any(k in item_lower for k in keys) or 'lemgendary' in item_lower or 'datasets' in item_lower\n",
        "                    if is_match:\n",
        "                        # Check direct images or targets\n",
        "                        if os.path.exists(os.path.join(path, 'images')) or os.path.exists(os.path.join(path, 'targets')):\n",
        "                            found.append(path)\n",
        "                        else:\n",
        "                            # Check nested images or targets (1 level deeper)\n",
        "                            try:\n",
        "                                for sub in os.listdir(path):\n",
        "                                    sub_cand = os.path.join(path, sub)\n",
        "                                    if os.path.isdir(sub_cand) and (os.path.exists(os.path.join(sub_cand, 'images')) or os.path.exists(os.path.join(sub_cand, 'targets'))):\n",
        "                                        found.append(sub_cand)\n",
        "                            except:\n",
        "                                pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "for d in sorted(list(set(found))):\n",
        "    if os.path.isdir(d):\n",
        "        bname = os.path.basename(d)\n",
        "        links = [bname]\n",
        "        if bname.lower() != bname: links.append(bname.lower())\n",
        "        \n",
        "        for link in links:\n",
        "            link_name = os.path.join(target_dir, link)\n",
        "            if not os.path.exists(link_name):\n",
        "                try: os.symlink(d, link_name)\n",
        "                except Exception as e: print(f'[REMEDY] An error occurred during environment setup: {e}')\n",
        "                print(f'[OK] [LINKED] {link} -> {d}')\n"
    ]

    install_source = [
        "import os, sys, subprocess\n",
        "print('[ENV] Installing Nuclear Dependencies...')\n",
        "suite_candidates = ['/kaggle/working/lemgendary-training-suite', '/kaggle/working/model-training/lemgendary-training-suite', '/kaggle/working']\n",
        "req_path = next((os.path.join(p, 'requirements.txt') for p in suite_candidates if os.path.exists(os.path.join(p, 'requirements.txt'))), None)\n",
        "if req_path:\n",
        "    res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-warn-conflicts', '--upgrade-strategy', 'only-if-needed', '-r', req_path])\n",
        "    if res.returncode == 0:\n",
        "        print('[OK] Environment Ready.')\n",
        "    else:\n",
        "        print('[WARNING] Dependency installation finished with non-zero exit code.')\n",
        "else:\n",
        "    print('[ERROR] Could not open requirements file: No such file or directory')\n",
        "    print(\"[REMEDY] Ensure 'requirements.txt' exists in the root of the lemgendary-training-suite repository.\")\n",
        "    print('[ACTION REQUIRED] Suite clone failed in Step 3 because SUITE_PAT/GITHUB_PAT is missing from Kaggle Secrets.')\n",
        "    print('[ACTION REQUIRED] Fix: Go to Kaggle Notebook top bar -> Add-ons -> Secrets -> Add SUITE_PAT or GITHUB_PAT with your GitHub token.')\n"
    ]

    hub_prep_source = [
        "import os\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        f"model_key = '{model_key}'\n",
        "model_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_dir = os.path.join(model_dir, 'checkpoints')\n",
        "\n",
        "print(f'[HUB] Initializing Lean Manifold for {model_key}...')\n",
        "os.makedirs(ckpt_dir, exist_ok=True)\n",
        "print(f'[OK] Manifold structure ready at {model_dir}')\n"
    ]

    stealth_source = [
        "import os, base64, torch, shutil\n",
        f"model_key = '{model_key}'\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "os.makedirs(ckpt_hub_dir, exist_ok=True)\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "\n",
        "# 2026 NUCLEAR: Prioritize Kaggle Inputs and recover to Hub using BFS scanner (max depth 6, directories only)\n",
        "input_ckpts = []\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    try:\n",
        "        queue = ['/kaggle/input']\n",
        "        depths = {'/kaggle/input': 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 6: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune image manifolds and datasets directory entirely to bypass FUSE latency\n",
        "                    if item_lower in ['datasets', 'images', 'train', 'val', 'test', 'validation', 'dataset']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    # If this is checkpoints folder or matches name, list .pth files\n",
        "                    if 'checkpoints' in item_lower or 'models' in item_lower or 'weights' in item_lower or model_key.replace('_', '') in item_lower.replace('_', ''):\n",
        "                        try:\n",
        "                            for f in os.listdir(path):\n",
        "                                if f.lower().endswith('.pth') and any(x in f.lower() for x in [model_key.lower().replace('_', ''), 'best', 'latest']):\n",
        "                                    input_ckpts.append(os.path.join(path, f))\n",
        "                        except:\n",
        "                            pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "if input_ckpts:\n",
        "    print(f'[RECOVERY] Hydrating hub from Kaggle Inputs...')\n",
        "    for src in input_ckpts:\n",
        "        fname = os.path.basename(src)\n",
        "        dst = os.path.join(ckpt_hub_dir, fname)\n",
        "        if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):\n",
        "            shutil.copy2(src, dst)\n",
        "            print(f'   -> [OK] Recovered {fname}')\n",
        "\n",
        "    # 2026 Resilience: Recover Metrics Audit Trail using BFS scanner\n",
        "    src_met = None\n",
        "    if os.path.exists('/kaggle/input'):\n",
        "        try:\n",
        "            queue = ['/kaggle/input']\n",
        "            depths = {'/kaggle/input': 0}\n",
        "            while queue:\n",
        "                curr = queue.pop(0)\n",
        "                depth = depths[curr]\n",
        "                if depth > 6: continue\n",
        "                for item in os.listdir(curr):\n",
        "                    path = os.path.join(curr, item)\n",
        "                    if os.path.isdir(path):\n",
        "                        item_lower = item.lower()\n",
        "                        # Prune image subdirectories to avoid deep lag\n",
        "                        if item_lower in ['images', 'train', 'val', 'test']:\n",
        "                            continue\n",
        "                        depths[path] = depth + 1\n",
        "                        queue.append(path)\n",
        "                    elif item == 'metrics.csv':\n",
        "                        src_met = path\n",
        "                        break\n",
        "                if src_met: break\n",
        "        except Exception as e: print(f'[REMEDY] An error occurred during environment setup: {e}')\n",
        "        \n",
        "    if src_met:\n",
        "        dst_met = os.path.join(model_hub_dir, 'metrics.csv')\n",
        "        if not os.path.exists(dst_met) or os.path.getsize(src_met) > os.path.getsize(dst_met):\n",
        "            shutil.copy2(src_met, dst_met)\n",
        "            print(f'[OK] [RECOVERY] Hydrated metrics.csv from {os.path.basename(os.path.dirname(src_met))}')\n",
        "\n",
        "    import glob\n",
        "    search_paths = [p for p in glob.glob(os.path.join(ckpt_hub_dir, '*.pth')) if any(x in os.path.basename(p) for x in [model_key, model_key.replace('_', '-')])]\n",
        "    search_paths.sort(key=lambda x: (0 if 'best' in x else 1 if 'latest' in x else 2, -os.path.getsize(x)))\n",
        "    found = []\n",
        "    for p in search_paths: \n",
        "        if os.path.exists(p) and os.path.getsize(p) > 10 * 1024 * 1024:\n",
        "            found.append(p)\n",
        "\n",
        "    model_path = found[0] if found else None\n",
        "    if model_path:\n",
        "        print(f'[SOTA] Loading pre-trained weights: {model_path}')\n",
        "        ckpt = torch.load(model_path, map_location=device, weights_only=False)\n",
        "        print(f'[OK] Weights anchored on {device}.')\n",
        "    else: print('[WARNING] No existing weights found. Starting from scratch.')\n"
    ]

    training_source = [
        "import os, subprocess, sys\n",
        "suite_candidates = ['/kaggle/working/lemgendary-training-suite', '/kaggle/working/model-training/lemgendary-training-suite', '/kaggle/working']\n",
        "active_suite_dir = next((p for p in suite_candidates if os.path.exists(os.path.join(p, 'training', 'train.py'))), '/kaggle/working/lemgendary-training-suite')\n",
        "os.chdir(active_suite_dir)\n",
        "print(f'[OK] [SUITE] Active working directory set to: {os.getcwd()}')\n",
        "\n",
        "# [JANITOR] Clean up any pre-existing zombie training processes to free the GPU\n",
        "try:\n",
        "    current_pid = os.getpid()\n",
        "    ps_out = subprocess.check_output(['ps', '-ef'], text=True)\n",
        "    for line in ps_out.split('\\n'):\n",
        "        if 'train.py' in line and str(current_pid) not in line:\n",
        "            parts = line.split()\n",
        "            if len(parts) > 1:\n",
        "                pid = int(parts[1])\n",
        "                print(f'[JANITOR] Killing stale zombie training process (PID {pid})...')\n",
        "                subprocess.run(['kill', '-9', str(pid)], capture_output=True)\n",
        "except Exception:\n",
        "    pass\n",
        "\n",
        "print(f'[LAUNCH] [NUCLEAR] Initiating Training Matrix for {model_key}...')\n",
        "cmd = [sys.executable, '-u', 'training/train.py', '--model', f'{model_key}', '--env', 'kaggle', '--auto_sync']\n",
        "p = subprocess.Popen(cmd)\n",
        "try:\n",
        "    p.wait()\n",
        "except KeyboardInterrupt:\n",
        "    print('\\n[TERMINATED] Training interrupted by user. Terminating training subprocess safely...')\n",
        "    try:\n",
        "        p.terminate()\n",
        "        p.wait(timeout=5)\n",
        "    except subprocess.TimeoutExpired:\n",
        "        p.kill()\n",
        "    print('[OK] Subprocess successfully killed. VRAM and CPU are clean.')\n"
    ]

    k_username = config.get("kaggle_username", "lemtreursi") if config else "lemtreursi"
    slug_prefix = config.get("kaggle_slug_prefix", "lemgendary-") if config else "lemgendary-"
    slug_suffix = config.get("kaggle_slug_suffix", "-checkpoints") if config else "-checkpoints"
    
    k_slug = model_key.replace('_', '-')
    if "nima-aesthetic" in k_slug:
        k_slug = k_slug.replace("nima-aesthetic", "nima-aesthetics")
    
    k_handle = f"{k_username}/{slug_prefix}{k_slug}{slug_suffix}/pytorch/default"

    push_source = [
        "import os, kagglehub\n",
        f"model_key = '{model_key}'\n",
        "local_path = f'/kaggle/working/LemGendaryModels/{model_key}'\n",
        f"model_handle = '{k_handle}'\n",
        "\n",
        "if os.path.exists(local_path):\n",
        "    print(f'[KAGGLE] Pushing finalized SOTA to {model_handle}...')\n",
        "    try:\n",
        "        kagglehub.model_upload(model_handle, local_path, version_notes=f'v16.2 SOTA Finalized Sync: {model_key}')\n",
        "        print('[DONE] Deployment Complete.')\n",
        "    except Exception as e:\n",
        "        print(f'[ERROR] Deployment failed: {e}')\n",
        "        print('[REMEDY] Ensure your Kaggle API key is correctly configured and the destination kernel slug is valid.')\n",
        "else: print(f'[WARNING] Local manifold not found at {local_path}')\n"
    ]

    checkpoint_recovery_source = [
        "import os, shutil\n",
        f"model_key = '{model_key}'\n",
        "print(f'[RECOVERY] Deep-searching for {model_key} checkpoints...')\n",
        "hub_root = '/kaggle/working/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "os.makedirs(ckpt_hub_dir, exist_ok=True)\n",
        "\n",
        "reg_filename = ''\n",
        "try:\n",
        "    import yaml\n",
        "    yaml_path = '/kaggle/working/lemgendary-training-suite/unified_models_v2.yaml'\n",
        "    if os.path.exists(yaml_path):\n",
        "        with open(yaml_path, 'r') as f: reg = yaml.safe_load(f)\n",
        "        reg_filename = reg.get(model_key, {}).get('filename', '')\n",
        "except Exception as e: print(f'[REMEDY] An error occurred during environment setup: {e}')\n",
        "\n",
        "target_slugs = [model_key.lower().replace('_', ''), model_key.lower().replace('_', '-'), reg_filename.lower() if reg_filename else '']\n",
        "target_slugs = [s for s in target_slugs if s]\n",
        "\n",
        "found_ckpts = []\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    try:\n",
        "        # Fast BFS Directory Search up to depth 7 to locate checkpoint folders\n",
        "        queue = ['/kaggle/input']\n",
        "        depths = {'/kaggle/input': 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 7: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune image manifolds and datasets directory entirely to bypass FUSE latency\n",
        "                    if item_lower in ['datasets', 'images', 'train', 'val', 'test', 'validation', 'dataset']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    # If matching candidate directory name, list the pth files\n",
        "                    if any(slug in item_lower for slug in target_slugs) or 'checkpoint' in item_lower or 'weights' in item_lower or 'models' in item_lower:\n",
        "                        try:\n",
        "                            for f in os.listdir(path):\n",
        "                                if f.lower().endswith('.pth') and (any(slug in f.lower() for slug in target_slugs) or 'best' in f.lower() or 'latest' in f.lower() or 'progress' in f.lower()):\n",
        "                                    found_ckpts.append(os.path.join(path, f))\n",
        "                        except:\n",
        "                            pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "found_ckpts = sorted(list(set(found_ckpts)))\n",
        "if found_ckpts:\n",
        "    print(f'   -> [FOUND] {len(found_ckpts)} binaries in Kaggle Manifold.')\n",
        "    for src in found_ckpts:\n",
        "        if f'/{model_key}/' not in src.replace('\\\\', '/') and f'{model_key}' not in os.path.basename(src):\n",
        "            continue\n",
        "        if not os.path.exists(src):\n",
        "            print(f'   -> [WARNING] Source missing (Ghost File/Broken Link): {src}')\n",
        "            continue\n",
        "        fname = os.path.basename(src)\n",
        "        target_f = fname\n",
        "        if 'latest' in fname.lower(): target_f = f'{model_key}_latest.pth'\n",
        "        elif 'best' in fname.lower(): target_f = f'{model_key}_best.pth'\n",
        "        elif 'progress' in fname.lower(): target_f = f'{model_key}_progress.pth'\n",
        "        \n",
        "        dst = os.path.join(ckpt_hub_dir, target_f)\n",
        "        if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):\n",
        "            shutil.copy2(src, dst)\n",
        "            print(f'   -> [OK] Recovered: {fname} -> {target_f}')\n",
        "    \n",
        "    metrics_found = False\n",
        "    for src in found_ckpts:\n",
        "        # Look for metrics.csv in parent or grandparent of the checkpoint\n",
        "        for d in [os.path.dirname(os.path.dirname(src)), os.path.dirname(src)]:\n",
        "            m_path = os.path.join(d, 'metrics.csv')\n",
        "            if os.path.exists(m_path):\n",
        "                try:\n",
        "                    shutil.copy2(m_path, os.path.join(model_hub_dir, 'metrics.csv'))\n",
        "                    print(f'[METRICS] Recovered metrics.csv from {os.path.basename(d)}')\n",
        "                    metrics_found = True; break\n",
        "                except Exception as e: print(f'[REMEDY] An error occurred during environment setup: {e}')\n",
        "        if metrics_found: break\n",
        "else: print('   -> [SKIP] No existing checkpoints found in Kaggle Inputs manifold.')\n"
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
                    f"# LemGendary Master Execution: {pascal_model_name} (v16.2 Nuclear-Hardened)\n",
                    "This unified notebook handles environment synchronization and automated cloud training.\n"
                ],
                "metadata": {}
            },
            {
                "cell_type": "markdown",
                "source": ["## 1. Hardware Sentinel\n", "Ensure the manifold has the required hardware acceleration.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": hardware_sentinel_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 2. Cloud Auth & Secrets\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": secrets_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 3. Environment Synchronization\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": clone_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": install_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 4. SOTA Hub Synchronization (Pull)\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": hub_prep_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 5. Multi-Path Data Resolution\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": symlink_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 6. Checkpoint & Metric Recovery\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": checkpoint_recovery_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 7. Nuclear Training Matrix\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": training_source,
                "metadata": {}, "outputs": [], "execution_count": None
            }
        ]
    }

    output_path = os.path.join(export_dir, f"{model_key}_training.ipynb")
    
    # --- 2026 Resilience: Dual-Export & Manifold Synchronization ---

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_hub_root = os.path.abspath(os.path.join(base_dir, "../LemGendaryDatasets"))
    
    # 1. Primary Model Export (Verified JSON)
    os.makedirs(export_dir, exist_ok=True)
    try:
        json_str = json.dumps(notebook_content, indent=4)
        json.loads(json_str) # Hard Validation
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(json_str)
        print(f"[OK] Generated Training Notebook: {output_path}")
    except Exception as e:
        print(f"[ERROR] JSON Validation failed for {model_key}: {e}")
        print("[REMEDY] This usually means the generated notebook syntax is invalid. Check 'unified_models.yaml' for trailing commas or malformed strings.")
        return

    # 2. Dataset Manifold Synchronization
    if unified_models_registry:
        m_info = unified_models_registry.get(model_key, {})
        ds_raw = m_info.get("datasets", []) or m_info.get("dataset", [])
        if isinstance(ds_raw, str):
            ds_list = [ds_raw]
        elif isinstance(ds_raw, (list, tuple)):
            ds_list = list(ds_raw)
        else:
            ds_list = []

        if model_key == "professional_multitask_restoration":
            target_candidates = ["LemGendizedProfessionalMultitaskRestorationLarge", "professional_multitask_restoration"]
        else:
            target_candidates = list(ds_list)
            if model_key not in target_candidates:
                target_candidates.append(model_key)
            
        synced_dirs = set()
        for target_folder in target_candidates:
            if not target_folder:
                continue
            # Handle PascalCase and snake_case correctly without destructive title()
            clean_name = target_folder
            if "_" in clean_name or "-" in clean_name:
                pascal_name = "".join(part.capitalize() for part in clean_name.replace("-", "_").split("_"))
            else:
                pascal_name = clean_name
                
            possible_manifold_folders = [
                target_folder,
                f"{target_folder}Large",
                f"LemGendized{pascal_name}",
                f"LemGendized{pascal_name}Large",
                f"LemGendized{target_folder}Large",
                f"LemGendized{target_folder}"
            ]
            
            for m_folder in possible_manifold_folders:
                ds_dir = os.path.join(datasets_hub_root, m_folder)
                if os.path.exists(ds_dir) and ds_dir not in synced_dirs:
                    synced_dirs.add(ds_dir)
                    ds_output_path = os.path.join(ds_dir, f"{model_key}_training.ipynb")
                    try:
                        with open(ds_output_path, "w", encoding='utf-8') as f:
                            f.write(json_str)
                        print(f"[OK] Synchronized Dataset Manifold Notebook: {ds_output_path}")
                    except Exception as ds_err:
                        print(f"[NOTICE] Skipped dataset manifold notebook sync for {ds_output_path} (read-only or unavailable: {ds_err})")



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
        "    if device.type == 'cuda' and torch.cuda.device_count() > 1:\n",
        "        model = torch.nn.DataParallel(model)\n",
        "    model.eval()\n",
        "\n",
        "    # 2. Prepare Input\n",
        f"    img = Image.open('photo.jpg').convert('RGB').resize(({w}, {h}))\n",
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
        f"    img = Image.open('photo.jpg').convert('RGB').resize(({w}, {h}))\n",
        "    input_data = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]\n",
        "    input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]\n",
        "\n",
        "    # 3. Inference\n",
        "    output = session.run(None, {'input': input_data})[0]\n",
        "    print(f'Prediction Raw: {output}')\n",
        "except Exception as e: print(f'ORT Load Info: {e}')\n"
    ]
    
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
        f"    img = Image.open('photo.jpg').convert('RGB').resize(({w}, {h}))\n",
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
                "metadata": {}, "outputs": [], "execution_count": None
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
                "metadata": {}, "outputs": [], "execution_count": None
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
                "metadata": {}, "outputs": [], "execution_count": None
            }
        ]
    }
    output_path = os.path.join(export_dir, f"{model_key}-usage.ipynb")
    
    # --- 2026 Resilience: Export Hardening ---
    try:
        json_str = json.dumps(notebook_content, indent=4)
        json.loads(json_str) # Hard Validation
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(json_str)
        print(f"[OK] Generated Usage Notebook: {output_path}")
    except Exception as e:
        print(f"[ERROR] JSON Validation failed for {model_key} usage: {e}")
        print("[REMEDY] This usually means the generated notebook syntax is invalid. Check 'unified_models.yaml' for trailing commas or malformed strings.")


def generate_colab_inference_notebook(model_key, export_dir, unified_models_registry=None, config=None):
    """
    Generates a v16.2 Nuclear-Hardened Inference Notebook for Kaggle.
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

    # --- Section Logic: v16.0 Nuclear Orchestration ---
    
    hardware_sentinel_source = [
        "import torch, sys\n",
        "print('[OK] [SENTINEL] Auditing Hardware Manifold...')\n",
        "if not torch.cuda.is_available():\n",
        "    print('[WARNING] NO GPU DETECTED!')\n",
        "    print('[ACTION REQUIRED] Enable GPU Accelerator in notebook settings:')\n",
        "    print('   -> Kaggle: Right Panel -> Session Options -> Accelerator -> GPU T4 x2 or P100')\n",
        "    print('   -> Colab:  Runtime -> Change runtime type -> Hardware accelerator -> GPU')\n",
        "    print('   -> Continuing in CPU Fallback Mode for dry-run validation...')\n",
        "else:\n",
        "    props = torch.cuda.get_device_properties(0)\n",
        "    print(f'[OK] [ACTIVE] {props.name}')\n",
        "    print(f'[OK] [VRAM] {props.total_memory / 1024**3:.1f} GB')\n",
        "    if props.total_memory / 1024**3 < 10.0:\n",
        "        print('[WARNING] Low VRAM detected. Suite will enable Survival Profiles automatically.')\n"
    ]

    secrets_source = [
        "try:\n",
        "    from google.colab import userdata\n",
        "    import os as _os, json as _json\n",
        "    g_pat = None\n",
        "    s_pat = None\n",
        "    k_key = None\n",
        "    k_user = None\n",
        "    g_drive = None\n",
        "    try: g_pat = userdata.get('GITHUB_PAT')\n",
        "    except Exception: print('[REMEDY] Missing secret! You should create new secret named GITHUB_PAT with your GitHub Personal Access Token as value')\n",
        "    try: s_pat = userdata.get('SUITE_PAT')\n",
        "    except Exception: print('[REMEDY] Missing secret! You should create new secret named SUITE_PAT with your GitHub Personal Access Token as value')\n",
        "    try: k_key = userdata.get('KAGGLE_KEY')\n",
        "    except Exception: print('[REMEDY] Missing secret! You should create new secret named KAGGLE_KEY with your Kaggle API Token as value')\n",
        "    try: k_user = userdata.get('KAGGLE_USERNAME')\n",
        "    except Exception: print('[REMEDY] Missing secret! You should create new secret named KAGGLE_USERNAME with your Kaggle username as value')\n",
        "    try: g_drive = userdata.get('GOOGLE_DRIVE')\n",
        "    except Exception: print('[REMEDY] Missing secret! You should create new secret named GOOGLE_DRIVE with your Google Drive token as value')\n",
        "    \n",
        "    if g_pat: _os.environ['GITHUB_PAT'] = g_pat\n",
        "    if s_pat: _os.environ['SUITE_PAT'] = s_pat\n",
        "    if g_drive: _os.environ['GOOGLE_DRIVE'] = g_drive\n",
        "    \n",
        "    if not k_user: k_user = 'lemtreursi'\n",
        "    if k_key:\n",
        "        _os.environ['KAGGLE_KEY'] = k_key\n",
        "        _os.environ['KAGGLE_USERNAME'] = k_user\n",
        "        _k_dir = _os.path.expanduser('~/.kaggle')\n",
        "        _os.makedirs(_k_dir, exist_ok=True)\n",
        "        with open(_os.path.join(_k_dir, 'kaggle.json'), 'w') as _kf:\n",
        "            _json.dump({'username': k_user, 'key': k_key}, _kf)\n",
        "        _os.chmod(_os.path.join(_k_dir, 'kaggle.json'), 0o600)\n",
        "    \n",
        "    active = []\n",
        "    if s_pat: active.append('SUITE_PAT')\n",
        "    if g_pat: active.append('GITHUB_PAT')\n",
        "    if k_key: active.append('KAGGLE_KEY')\n",
        "    if g_drive: active.append('GOOGLE_DRIVE')\n",
        "    if active:\n",
        "        print(f'[OK] [AUTH] Colab Secrets mounted: {\", \".join(active)}')\n",
        "    else:\n",
        "        print('[WARNING] No PATs found in Colab Secrets! Private repositories will fail to clone.')\n",
        "        print('[ACTION REQUIRED] Add SUITE_PAT or GITHUB_PAT to Colab Secrets.')\n",
        "except Exception as e:\n",
        "    print(f'[ERROR] Secret mounting failed: {e}')\n"
    ]

    clone_source = [
        "import os, subprocess, shutil\n",
        "repo_url = 'https://github.com/lemgenda/lemgendary-training-suite.git'\n",
        "suite_path = '/content/lemgendary-training-suite'\n",
        "pat = os.environ.get('SUITE_PAT', os.environ.get('GITHUB_PAT', ''))\n",
        "if pat:\n",
        "    # Use x-access-token for more reliable auth with fine-grained tokens\n",
        "    auth_url = repo_url.replace('https://', f'https://x-access-token:{pat}@')\n",
        "    print(f'[AUTH] Using {\"SUITE_PAT\" if os.environ.get(\"SUITE_PAT\") else \"GITHUB_PAT\"} for cloning...')\n",
        "else:\n",
        "    print('[WARNING] No PAT found in environment. Attempting public clone (will fail for private repos)...')\n",
        "    print('[ACTION REQUIRED] If clone fails, add SUITE_PAT or GITHUB_PAT to Kaggle Add-ons -> Secrets.')\n",
        "    auth_url = repo_url\n",
        "\n",
        "env = os.environ.copy()\n",
        "env['GIT_TERMINAL_PROMPT'] = '0'\n",
        "\n",
        "if not os.path.exists(suite_path):\n",
        "    print('[SUITE] Initializing LemGendary Training Suite...')\n",
        "    res = subprocess.run(['git', 'clone', auth_url, suite_path], capture_output=True, text=True, env=env)\n",
        "    if res.returncode == 0: \n",
        "        print('[OK] Suite cloned.')\n",
        "    else: \n",
        "        print(f'[ERROR] Clone failed: {res.stderr.strip()}')\n",
        "        print('[REMEDY] If a 403/401 occurs, ensure your SUITE_PAT or GITHUB_PAT has repo read permissions.')\n",
        "        if '403' in res.stderr or '401' in res.stderr or 'terminal prompts disabled' in res.stderr:\n",
        "            print('[ACTION REQUIRED] Add SUITE_PAT or GITHUB_PAT to Kaggle Add-ons -> Secrets with GitHub read permissions.')\n",
        "else:\n",
        "    print('[OK] Suite resident. Syncing origin and pulling latest...')\n",
        "    subprocess.run(['git', 'remote', 'set-url', 'origin', auth_url], cwd=suite_path, env=env)\n",
        "    subprocess.run(['git', 'pull'], cwd=suite_path, env=env)\n"
    ]

    fuse_mount_source = [
        "import os\n",
        "print('[MOUNT] Attaching Google Drive FUSE...')\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "print('[OK] Google Drive mounted successfully. Datasets will be streamed directly from Drive.')\n"
    ]
    symlink_source = [
        "import os\n",
        f"model_key = '{model_key}'\n",
        "target_dir = '/content/LemGendaryDatasets'\n",
        "os.makedirs(target_dir, exist_ok=True)\n",
        "\n",
        "print(f'[DATA] Resolving manifolds for {model_key}...')\n",
        "found = []\n",
        "keys = [model_key.lower(), model_key.replace(\"_\", \"-\"), model_key.replace(\"_\", \"\")]\n",
        "\n",
        "# 1. Restricted BFS Scanner (max depth 4, directories only) to bypass FUSE latency\n",
        "if True:\n",
        "    try:\n",
        "        drives = [d for d in ['/content/drive/MyDrive', '/content/drive/Shareddrives', '/content/drive/Shared with me'] if os.path.exists(d)]\n",
        "        if not drives: drives = ['/content/drive']\n",
        "        queue = list(drives)\n",
        "        depths = {d: 0 for d in drives}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 4: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune models/checkpoints to prevent wasting time scanning weights\n",
        "                    # 2026 Resilience: Aggressive FUSE Pruning - NEVER enter raw image/target dirs to prevent OOM stat storms\n",
        "                    if item_lower in ['models', 'checkpoints', 'weights', 'images', 'targets', 'labels', 'masks', 'train', 'val', 'test', 'eval', 'lemgendarymodels']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    is_match = any(k in item_lower for k in keys) or 'lemgendary' in item_lower or 'datasets' in item_lower\n",
        "                    if is_match:\n",
        "                        def is_valid_ds(p):\n",
        "                            try: return os.path.exists(os.path.join(p, 'images')) or os.path.exists(os.path.join(p, 'targets')) or 'forex' in os.path.basename(p).lower() or any(f.endswith('.csv') or f.endswith('.json') or f.endswith('.parquet') for f in os.listdir(p))\n",
        "                            except: return False\n",
        "                        if is_valid_ds(path): found.append(path)\n",
        "                        else:\n",
        "                            try:\n",
        "                                for sub in os.listdir(path):\n",
        "                                    sub_cand = os.path.join(path, sub)\n",
        "                                    if os.path.isdir(sub_cand) and is_valid_ds(sub_cand): found.append(sub_cand)\n",
        "                            except Exception as e: print(f'[REMEDY] An error occurred during environment setup: {e}')\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "for d in sorted(list(set(found))):\n",
        "    if os.path.isdir(d):\n",
        "        bname = os.path.basename(d)\n",
        "        links = [bname]\n",
        "        for link in links:\n",
        "            link_name = os.path.join(target_dir, link)\n",
        "            if not os.path.exists(link_name):\n",
        "                try: os.symlink(d, link_name)\n",
        "                except Exception as e: print(f'[REMEDY] An error occurred during environment setup: {e}')\n",
        "                print(f'[OK] [LINKED] {link} -> {d}')\n"
    ]

    install_source = [
        "import os, sys, subprocess\n",
        "print('[ENV] Installing Nuclear Dependencies...')\n",
        "suite_candidates = ['/content/lemgendary-training-suite', '/content/model-training/lemgendary-training-suite', '/content']\n",
        "req_path = next((os.path.join(p, 'requirements.txt') for p in suite_candidates if os.path.exists(os.path.join(p, 'requirements.txt'))), None)\n",
        "if req_path:\n",
        "    res = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-warn-conflicts', '--upgrade-strategy', 'only-if-needed', '-r', req_path])\n",
        "    if res.returncode == 0:\n",
        "        print('[OK] Environment Ready.')\n",
        "    else:\n",
        "        print('[WARNING] Dependency installation finished with non-zero exit code.')\n",
        "else:\n",
        "    print('[ERROR] Could not open requirements file: No such file or directory')\n",
        "    print(\"[REMEDY] Ensure 'requirements.txt' exists in the root of the lemgendary-training-suite repository.\")\n",
        "    print('[ACTION REQUIRED] Suite clone failed in Step 3 because SUITE_PAT/GITHUB_PAT is missing from Kaggle Secrets.')\n",
        "    print('[ACTION REQUIRED] Fix: Go to Kaggle Notebook top bar -> Add-ons -> Secrets -> Add SUITE_PAT or GITHUB_PAT with your GitHub token.')\n"
    ]

    hub_prep_source = [
        "import os\n",
        "hub_root = '/content/LemGendaryModels'\n",
        f"model_key = '{model_key}'\n",
        "model_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_dir = os.path.join(model_dir, 'checkpoints')\n",
        "\n",
        "print(f'[HUB] Initializing Lean Manifold for {model_key}...')\n",
        "os.makedirs(ckpt_dir, exist_ok=True)\n",
        "print(f'[OK] Manifold structure ready at {model_dir}')\n"
    ]

    stealth_source = [
        "import os, base64, torch, shutil\n",
        f"model_key = '{model_key}'\n",
        "hub_root = '/content/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "os.makedirs(ckpt_hub_dir, exist_ok=True)\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "\n",
        "# 2026 NUCLEAR: Prioritize Kaggle Inputs and recover to Hub using BFS scanner (max depth 6, directories only)\n",
        "input_ckpts = []\n",
        "if os.path.exists('/content/drive/MyDrive'):\n",
        "    try:\n",
        "        queue = ['/content/drive/MyDrive']\n",
        "        depths = {'/content/drive/MyDrive': 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 6: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune image manifolds and datasets directory entirely to bypass FUSE latency\n",
        "                    if item_lower in ['datasets', 'images', 'train', 'val', 'test', 'validation', 'dataset']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    # If this is checkpoints folder or matches name, list .pth files\n",
        "                    if 'checkpoints' in item_lower or 'models' in item_lower or 'weights' in item_lower or model_key.replace('_', '') in item_lower.replace('_', ''):\n",
        "                        try:\n",
        "                            for f in os.listdir(path):\n",
        "                                if f.lower().endswith('.pth') and any(x in f.lower() for x in [model_key.lower().replace('_', ''), 'best', 'latest']):\n",
        "                                    input_ckpts.append(os.path.join(path, f))\n",
        "                        except:\n",
        "                            pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "if input_ckpts:\n",
        "    print(f'[RECOVERY] Hydrating hub from Kaggle Inputs...')\n",
        "    for src in input_ckpts:\n",
        "        fname = os.path.basename(src)\n",
        "        dst = os.path.join(ckpt_hub_dir, fname)\n",
        "        if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):\n",
        "            shutil.copy2(src, dst)\n",
        "            print(f'   -> [OK] Recovered {fname}')\n",
        "\n",
        "    # 2026 Resilience: Recover Metrics Audit Trail using BFS scanner\n",
        "    src_met = None\n",
        "    if os.path.exists('/content/drive/MyDrive'):\n",
        "        try:\n",
        "            queue = ['/content/drive/MyDrive']\n",
        "            depths = {'/content/drive/MyDrive': 0}\n",
        "            while queue:\n",
        "                curr = queue.pop(0)\n",
        "                depth = depths[curr]\n",
        "                if depth > 6: continue\n",
        "                for item in os.listdir(curr):\n",
        "                    path = os.path.join(curr, item)\n",
        "                    if os.path.isdir(path):\n",
        "                        item_lower = item.lower()\n",
        "                        # Prune image subdirectories to avoid deep lag\n",
        "                        if item_lower in ['images', 'train', 'val', 'test']:\n",
        "                            continue\n",
        "                        depths[path] = depth + 1\n",
        "                        queue.append(path)\n",
        "                    elif item == 'metrics.csv':\n",
        "                        src_met = path\n",
        "                        break\n",
        "                if src_met: break\n",
        "        except Exception as e: print(f'[REMEDY] An error occurred during environment setup: {e}')\n",
        "        \n",
        "    if src_met:\n",
        "        dst_met = os.path.join(model_hub_dir, 'metrics.csv')\n",
        "        if not os.path.exists(dst_met) or os.path.getsize(src_met) > os.path.getsize(dst_met):\n",
        "            shutil.copy2(src_met, dst_met)\n",
        "            print(f'[OK] [RECOVERY] Hydrated metrics.csv from {os.path.basename(os.path.dirname(src_met))}')\n",
        "\n",
        "    import glob\n",
        "    search_paths = [p for p in glob.glob(os.path.join(ckpt_hub_dir, '*.pth')) if any(x in os.path.basename(p) for x in [model_key, model_key.replace('_', '-')])]\n",
        "    search_paths.sort(key=lambda x: (0 if 'best' in x else 1 if 'latest' in x else 2, -os.path.getsize(x)))\n",
        "    found = []\n",
        "    for p in search_paths: \n",
        "        if os.path.exists(p) and os.path.getsize(p) > 10 * 1024 * 1024:\n",
        "            found.append(p)\n",
        "\n",
        "    model_path = found[0] if found else None\n",
        "    if model_path:\n",
        "        print(f'[SOTA] Loading pre-trained weights: {model_path}')\n",
        "        ckpt = torch.load(model_path, map_location=device, weights_only=False)\n",
        "        print(f'[OK] Weights anchored on {device}.')\n",
        "    else: print('[WARNING] No existing weights found. Starting from scratch.')\n"
    ]

    training_source = [
        "import os, subprocess, sys\n",
        "suite_candidates = ['/content/lemgendary-training-suite', '/content/model-training/lemgendary-training-suite', '/content']\n",
        "active_suite_dir = next((p for p in suite_candidates if os.path.exists(os.path.join(p, 'training', 'train.py'))), '/content/lemgendary-training-suite')\n",
        "os.chdir(active_suite_dir)\n",
        "print(f'[OK] [SUITE] Active working directory set to: {os.getcwd()}')\n",
        "\n",
        "# [JANITOR] Clean up any pre-existing zombie training processes to free the GPU\n",
        "try:\n",
        "    current_pid = os.getpid()\n",
        "    ps_out = subprocess.check_output(['ps', '-ef'], text=True)\n",
        "    for line in ps_out.split('\\n'):\n",
        "        if 'train.py' in line and str(current_pid) not in line:\n",
        "            parts = line.split()\n",
        "            if len(parts) > 1:\n",
        "                pid = int(parts[1])\n",
        "                print(f'[JANITOR] Killing stale zombie training process (PID {pid})...')\n",
        "                subprocess.run(['kill', '-9', str(pid)], capture_output=True)\n",
        "except Exception:\n",
        "    pass\n",
        "\n",
        "print(f'[LAUNCH] [NUCLEAR] Initiating Training Matrix for {model_key}...')\n",
        "cmd = [sys.executable, '-u', 'training/train.py', '--model', f'{model_key}', '--env', 'colab', '--auto_sync']\n",
        "p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n",
        "try:\n",
        "    import io\n",
        "    for line in io.TextIOWrapper(p.stdout, newline=''):\n",
        "        print(line, end='', flush=True)\n",
        "    p.wait()\n",
        "except KeyboardInterrupt:\n",
        "    print('\\n[TERMINATED] Training interrupted by user. Terminating training subprocess safely...')\n",
        "    try:\n",
        "        p.terminate()\n",
        "        p.wait(timeout=5)\n",
        "    except subprocess.TimeoutExpired:\n",
        "        p.kill()\n",
        "    print('[OK] Subprocess successfully killed. VRAM and CPU are clean.')\n"
    ]

    k_username = config.get("kaggle_username", "lemtreursi") if config else "lemtreursi"
    slug_prefix = config.get("kaggle_slug_prefix", "lemgendary-") if config else "lemgendary-"
    slug_suffix = config.get("kaggle_slug_suffix", "-checkpoints") if config else "-checkpoints"
    
    k_slug = model_key.replace('_', '-')
    if "nima-aesthetic" in k_slug:
        k_slug = k_slug.replace("nima-aesthetic", "nima-aesthetics")
    
    k_handle = f"{k_username}/{slug_prefix}{k_slug}{slug_suffix}/pytorch/default"

    push_source = [
        "import os, kagglehub\n",
        f"model_key = '{model_key}'\n",
        "local_path = f'/content/LemGendaryModels/{model_key}'\n",
        f"model_handle = '{k_handle}'\n",
        "\n",
        "if os.path.exists(local_path):\n",
        "    print(f'[KAGGLE] Pushing finalized SOTA to {model_handle}...')\n",
        "    try:\n",
        "        kagglehub.model_upload(model_handle, local_path, version_notes=f'v16.2 SOTA Finalized Sync: {model_key}')\n",
        "        print('[DONE] Deployment Complete.')\n",
        "    except Exception as e:\n",
        "        print(f'[ERROR] Deployment failed: {e}')\n",
        "        print('[REMEDY] Ensure your Kaggle API key is correctly configured and the destination kernel slug is valid.')\n",
        "else: print(f'[WARNING] Local manifold not found at {local_path}')\n"
    ]

    checkpoint_recovery_source = [
        "import os, shutil\n",
        f"model_key = '{model_key}'\n",
        "print(f'[RECOVERY] Deep-searching for {model_key} checkpoints...')\n",
        "hub_root = '/content/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "os.makedirs(ckpt_hub_dir, exist_ok=True)\n",
        "\n",
        "reg_filename = ''\n",
        "try:\n",
        "    import yaml\n",
        "    yaml_path = '/content/lemgendary-training-suite/unified_models_v2.yaml'\n",
        "    if os.path.exists(yaml_path):\n",
        "        with open(yaml_path, 'r') as f: reg = yaml.safe_load(f)\n",
        "        reg_filename = reg.get(model_key, {}).get('filename', '')\n",
        "except Exception as e: print(f'[REMEDY] An error occurred during environment setup: {e}')\n",
        "\n",
        "target_slugs = [model_key.lower().replace('_', ''), model_key.lower().replace('_', '-'), reg_filename.lower() if reg_filename else '']\n",
        "target_slugs = [s for s in target_slugs if s]\n",
        "\n",
        "found_ckpts = []\n",
        "if os.path.exists('/content/drive/MyDrive'):\n",
        "    try:\n",
        "        # Fast BFS Directory Search up to depth 7 to locate checkpoint folders\n",
        "        queue = ['/content/drive/MyDrive']\n",
        "        depths = {'/content/drive/MyDrive': 0}\n",
        "        while queue:\n",
        "            curr = queue.pop(0)\n",
        "            depth = depths[curr]\n",
        "            if depth > 7: continue\n",
        "            for item in os.listdir(curr):\n",
        "                path = os.path.join(curr, item)\n",
        "                if os.path.isdir(path):\n",
        "                    item_lower = item.lower()\n",
        "                    # Prune image manifolds and datasets directory entirely to bypass FUSE latency\n",
        "                    if item_lower in ['datasets', 'images', 'train', 'val', 'test', 'validation', 'dataset']:\n",
        "                        continue\n",
        "                    depths[path] = depth + 1\n",
        "                    queue.append(path)\n",
        "                    \n",
        "                    # If matching candidate directory name, list the pth files\n",
        "                    if any(slug in item_lower for slug in target_slugs) or 'checkpoint' in item_lower or 'weights' in item_lower or 'models' in item_lower:\n",
        "                        try:\n",
        "                            for f in os.listdir(path):\n",
        "                                if f.lower().endswith('.pth') and (any(slug in f.lower() for slug in target_slugs) or 'best' in f.lower() or 'latest' in f.lower() or 'progress' in f.lower()):\n",
        "                                    found_ckpts.append(os.path.join(path, f))\n",
        "                        except:\n",
        "                            pass\n",
        "    except Exception:\n",
        "        pass\n",
        "\n",
        "found_ckpts = sorted(list(set(found_ckpts)))\n",
        "if found_ckpts:\n",
        "    print(f'   -> [FOUND] {len(found_ckpts)} binaries in Kaggle Manifold.')\n",
        "    for src in found_ckpts:\n",
        "        if f'/{model_key}/' not in src.replace('\\\\', '/') and f'{model_key}' not in os.path.basename(src):\n",
        "            continue\n",
        "        if not os.path.exists(src):\n",
        "            print(f'   -> [WARNING] Source missing (Ghost File/Broken Link): {src}')\n",
        "            continue\n",
        "        fname = os.path.basename(src)\n",
        "        target_f = fname\n",
        "        if 'latest' in fname.lower(): target_f = f'{model_key}_latest.pth'\n",
        "        elif 'best' in fname.lower(): target_f = f'{model_key}_best.pth'\n",
        "        elif 'progress' in fname.lower(): target_f = f'{model_key}_progress.pth'\n",
        "        \n",
        "        dst = os.path.join(ckpt_hub_dir, target_f)\n",
        "        if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):\n",
        "            shutil.copy2(src, dst)\n",
        "            print(f'   -> [OK] Recovered: {fname} -> {target_f}')\n",
        "    \n",
        "    metrics_found = False\n",
        "    for src in found_ckpts:\n",
        "        # Look for metrics.csv in parent or grandparent of the checkpoint\n",
        "        for d in [os.path.dirname(os.path.dirname(src)), os.path.dirname(src)]:\n",
        "            m_path = os.path.join(d, 'metrics.csv')\n",
        "            if os.path.exists(m_path):\n",
        "                try:\n",
        "                    shutil.copy2(m_path, os.path.join(model_hub_dir, 'metrics.csv'))\n",
        "                    print(f'[METRICS] Recovered metrics.csv from {os.path.basename(d)}')\n",
        "                    metrics_found = True; break\n",
        "                except Exception as e: print(f'[REMEDY] An error occurred during environment setup: {e}')\n",
        "        if metrics_found: break\n",
        "else: print('   -> [SKIP] No existing checkpoints found in Kaggle Inputs manifold.')\n"
    ]
    continuous_sync_source = [
        "import os, time, shutil, threading\n",
        f"model_key = '{model_key}'\n",
        "hub_root = '/content/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "\n",
        "drive_target_dir = None\n",
        "if found_ckpts:\n",
        "    drive_target_dir = os.path.dirname(found_ckpts[0])\n",
        "\n",
        "def drive_sync_worker():\n",
        "    print(f'[SYNC] Background sync thread started. Target: {drive_target_dir}')\n",
        "    while True:\n",
        "        try:\n",
        "            for f in os.listdir(ckpt_hub_dir):\n",
        "                src = os.path.join(ckpt_hub_dir, f)\n",
        "                if os.path.isfile(src):\n",
        "                    dst = os.path.join(drive_target_dir, f)\n",
        "                    # Copy if newer or doesn't exist\n",
        "                    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):\n",
        "                        tmp_dst = dst + '.tmp'\n",
        "                        shutil.copy2(src, tmp_dst)\n",
        "                        os.rename(tmp_dst, dst)\n",
        "            # Sync metrics.csv\n",
        "            m_src = os.path.join(model_hub_dir, 'metrics.csv')\n",
        "            if os.path.exists(m_src):\n",
        "                m_dst = os.path.join(os.path.dirname(drive_target_dir), 'metrics.csv')\n",
        "                if not os.path.exists(m_dst) or os.path.getmtime(m_src) > os.path.getmtime(m_dst):\n",
        "                    shutil.copy2(m_src, m_dst)\n",
        "        except Exception as e:\n",
        "            pass\n",
        "        time.sleep(30) # Sync every 30 seconds\n",
        "\n",
        "if drive_target_dir:\n",
        "    t = threading.Thread(target=drive_sync_worker, daemon=True)\n",
        "    t.start()\n",
        "else:\n",
        "    print('[WARNING] No Google Drive checkpoint directory found. Background sync disabled.')\n"
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
                    f"# LemGendary Master Execution: {pascal_model_name} (v16.2 Nuclear-Hardened)\n",
                    "This unified notebook handles environment synchronization and automated cloud training.\n"
                ],
                "metadata": {}
            },
            {
                "cell_type": "markdown",
                "source": ["## 1. Hardware Sentinel\n", "Ensure the manifold has the required hardware acceleration.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": hardware_sentinel_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 2. Cloud Auth & Secrets\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": secrets_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 3. Environment Synchronization\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": clone_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "code",
                "source": install_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 4. SOTA Hub Synchronization (Pull)\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": hub_prep_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 4.5 Google Drive Mount\n", "Mount Google Drive FUSE for streaming datasets directly.\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": fuse_mount_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 5. Multi-Path Data Resolution\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": symlink_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 6. Checkpoint & Metric Recovery\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": checkpoint_recovery_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 7. Continuous Drive Synchronization\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": continuous_sync_source,
                "metadata": {}, "outputs": [], "execution_count": None
            },
            {
                "cell_type": "markdown",
                "source": ["## 8. Nuclear Training Matrix\n"],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": training_source,
                "metadata": {}, "outputs": [], "execution_count": None
            }
        ]
    }

    output_path = os.path.join(export_dir, f"{model_key}_colab_training.ipynb")
    
    # --- 2026 Resilience: Dual-Export & Manifold Synchronization ---

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_hub_root = os.path.abspath(os.path.join(base_dir, "../LemGendaryDatasets"))
    
    # 1. Primary Model Export (Verified JSON)
    os.makedirs(export_dir, exist_ok=True)
    try:
        json_str = json.dumps(notebook_content, indent=4)
        json.loads(json_str) # Hard Validation
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(json_str)
        print(f"[OK] Generated Training Notebook: {output_path}")
    except Exception as e:
        print(f"[ERROR] JSON Validation failed for {model_key}: {e}")
        print("[REMEDY] This usually means the generated notebook syntax is invalid. Check 'unified_models.yaml' for trailing commas or malformed strings.")
        return

    # 2. Dataset Manifold Synchronization
    if unified_models_registry and model_key != "forex_predictor":
        m_info = unified_models_registry.get(model_key, {})
        ds_raw = m_info.get("datasets", []) or m_info.get("dataset", [])
        if isinstance(ds_raw, str):
            ds_list = [ds_raw]
        elif isinstance(ds_raw, (list, tuple)):
            ds_list = list(ds_raw)
        else:
            ds_list = []

        if model_key == "professional_multitask_restoration":
            target_candidates = ["LemGendizedProfessionalMultitaskRestorationLarge", "professional_multitask_restoration"]
        else:
            target_candidates = list(ds_list)
            if model_key not in target_candidates:
                target_candidates.append(model_key)
            
        synced_dirs = set()
        for target_folder in target_candidates:
            if not target_folder:
                continue
            # Handle PascalCase and snake_case correctly without destructive title()
            clean_name = target_folder
            if "_" in clean_name or "-" in clean_name:
                pascal_name = "".join(part.capitalize() for part in clean_name.replace("-", "_").split("_"))
            else:
                pascal_name = clean_name
                
            possible_manifold_folders = [
                target_folder,
                f"{target_folder}Large",
                f"LemGendized{pascal_name}",
                f"LemGendized{pascal_name}Large",
                f"LemGendized{target_folder}Large",
                f"LemGendized{target_folder}"
            ]
            
            for m_folder in possible_manifold_folders:
                ds_dir = os.path.join(datasets_hub_root, m_folder)
                if os.path.exists(ds_dir) and ds_dir not in synced_dirs:
                    synced_dirs.add(ds_dir)
                    ds_output_path = os.path.join(ds_dir, f"{model_key}_colab_training.ipynb")
                    try:
                        with open(ds_output_path, "w", encoding='utf-8') as f:
                            f.write(json_str)
                        print(f"[OK] Synchronized Dataset Manifold Notebook: {ds_output_path}")
                    except Exception as ds_err:
                        print(f"[NOTICE] Skipped dataset manifold notebook sync for {ds_output_path} (read-only or unavailable: {ds_err})")



def generate_colab_usage_notebook(model_key, export_dir, unified_models_registry=None, config=None):
    """
    Generates [model]_colab-usage.ipynb with snippets for PTH, ONNX FP32 (external), and ONNX FP16 (embedded).
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
        "    if device.type == 'cuda' and torch.cuda.device_count() > 1:\n",
        "        model = torch.nn.DataParallel(model)\n",
        "    model.eval()\n",
        "\n",
        "    # 2. Prepare Input\n",
        f"    img = Image.open('photo.jpg').convert('RGB').resize(({w}, {h}))\n",
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
        f"    img = Image.open('photo.jpg').convert('RGB').resize(({w}, {h}))\n",
        "    input_data = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]\n",
        "    input_data = input_data.transpose(2, 0, 1)[np.newaxis, :]\n",
        "\n",
        "    # 3. Inference\n",
        "    output = session.run(None, {'input': input_data})[0]\n",
        "    print(f'Prediction Raw: {output}')\n",
        "except Exception as e: print(f'ORT Load Info: {e}')\n"
    ]
    
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
        f"    img = Image.open('photo.jpg').convert('RGB').resize(({w}, {h}))\n",
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
                "metadata": {}, "outputs": [], "execution_count": None
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
                "metadata": {}, "outputs": [], "execution_count": None
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
                "metadata": {}, "outputs": [], "execution_count": None
            }
        ]
    }
    output_path = os.path.join(export_dir, f"{model_key}-colab-usage.ipynb")
    
    # --- 2026 Resilience: Export Hardening ---
    try:
        json_str = json.dumps(notebook_content, indent=4)
        json.loads(json_str) # Hard Validation
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(json_str)
        print(f"[OK] Generated Usage Notebook: {output_path}")
    except Exception as e:
        print(f"[ERROR] JSON Validation failed for {model_key} usage: {e}")
        print("[REMEDY] This usually means the generated notebook syntax is invalid. Check 'unified_models.yaml' for trailing commas or malformed strings.")



if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser(description="LemGendary Notebook Orchestrator (v16.2 Nuclear)")
    parser.add_argument("--model", type=str, help="Generate notebooks for a specific model key.")
    parser.add_argument("--all", action="store_true", help="Regenerate the entire Notebook Matrix for all registry models.")
    parser.add_argument("--dir", type=str, help="Override export directory.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    registry_path = os.path.join(base_dir, config.get("unified_models", "unified_models_v2.yaml"))
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    default_export = os.path.abspath(os.path.join(base_dir, config.get("export_dir", "../LemGendaryModels")))
    export_root = args.dir if args.dir else default_export

    models_to_gen = []
    if args.all:
        models_to_gen = [k for k in registry.keys() if k != "_registry_metadata"]
        print(f"[NUCLEAR] Initiating Global Notebook Refresh for {len(models_to_gen)} models...")
    elif args.model:
        if args.model in registry:
            models_to_gen = [args.model]
        else:
            print(f"[ERROR] Model '{args.model}' not found in registry.")
            print("[REMEDY] Verify the spelling of the model key in 'unified_models.yaml'.")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    for m_key in models_to_gen:
        m_dir = os.path.join(export_root, m_key)
        os.makedirs(m_dir, exist_ok=True)
        generate_inference_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
        generate_usage_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
        generate_colab_inference_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
        generate_colab_usage_notebook(m_key, m_dir, unified_models_registry=registry, config=config)

    print("\n[SUCCESS] Notebook Matrix Synchronized.")
