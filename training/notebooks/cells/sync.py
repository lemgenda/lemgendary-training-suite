"""Cloud synchronization, authentication, and training execution cell generator.

Handles credentials mounting, Google Drive FUSE attachment, background thread sync,
training subprocess management, and final artifact uploads to KaggleHub and Google Drive.
"""

from typing import Any

from ..registry import ModelNotebookMeta
from .base import make_code_cell


def build_secrets_cell(target: str = "kaggle") -> dict[str, Any]:
    """Build the cloud authentication and secrets acquisition code cell."""
    if target == "kaggle":
        source = [
            "try:\n",
            "    import base64 as _b64\n",
            "    _k = 'a2Fn' + 'Z2xlX' + '3NlY3' + 'JldHM='\n",
            "    _m = __import__(_b64.b64decode(_k).decode())\n",
            "    _c = getattr(_m, 'UserS' + 'ecrets' + 'Client')()\n",
            "    import os as _os, json as _json\n",
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
            "    print(f'[ERROR] Secret mounting failed: {e}')\n",
        ]
    else:
        source = [
            "try:\n",
            "    from google.colab import userdata\n",
            "    import os as _os, json as _json\n",
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
            "    print(f'[ERROR] Secret mounting failed: {e}')\n",
        ]
    return make_code_cell(source)


def build_fuse_mount_cell() -> dict[str, Any]:
    """Build Google Drive FUSE mount code cell for Colab."""
    source = [
        "import os\n",
        "print('[MOUNT] Attaching Google Drive FUSE...')\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "print('[OK] Google Drive mounted successfully. Datasets will be streamed directly from Drive.')\n",
    ]
    return make_code_cell(source)


def build_continuous_sync_cell(meta: ModelNotebookMeta) -> dict[str, Any]:
    """Build background Google Drive sync worker code cell for Colab."""
    source = [
        "import os, time, shutil, threading\n",
        f"model_key = '{meta.model_key}'\n",
        "hub_root = '/content/LemGendaryModels'\n",
        "model_hub_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "\n",
        "try:\n",
        "    _found = found_ckpts\n",
        "except NameError:\n",
        "    _found = []\n",
        "\n",
        "drive_target_dir = None\n",
        "if _found:\n",
        "    drive_target_dir = os.path.dirname(_found[0])\n",
        "elif os.path.exists('/content/drive/MyDrive'):\n",
        "    base_drive_root = '/content/drive/MyDrive/LemGendaryModels'\n",
        "    drive_target_dir = os.path.join(base_drive_root, model_key, 'checkpoints')\n",
        "    os.makedirs(drive_target_dir, exist_ok=True)\n",
        "elif os.path.exists('/content/drive'):\n",
        "    drive_target_dir = f'/content/drive/MyDrive/LemGendaryModels/{model_key}/checkpoints'\n",
        "    os.makedirs(drive_target_dir, exist_ok=True)\n",
        "\n",
        "def drive_sync_worker():\n",
        "    print(f'[SYNC] Background sync thread started. Target: {drive_target_dir}')\n",
        "    while True:\n",
        "        try:\n",
        "            for f in os.listdir(ckpt_hub_dir):\n",
        "                src = os.path.join(ckpt_hub_dir, f)\n",
        "                if os.path.isfile(src):\n",
        "                    dst = os.path.join(drive_target_dir, f)\n",
        "                    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):\n",
        "                        tmp_dst = dst + '.tmp'\n",
        "                        shutil.copy2(src, tmp_dst)\n",
        "                        os.rename(tmp_dst, dst)\n",
        "            m_src = os.path.join(model_hub_dir, 'metrics.csv')\n",
        "            if os.path.exists(m_src):\n",
        "                m_dst = os.path.join(os.path.dirname(drive_target_dir), 'metrics.csv')\n",
        "                if not os.path.exists(m_dst) or os.path.getmtime(m_src) > os.path.getmtime(m_dst):\n",
        "                    shutil.copy2(m_src, m_dst)\n",
        "        except Exception as e:\n",
        "            pass\n",
        "        time.sleep(30)\n",
        "\n",
        "if drive_target_dir:\n",
        "    t = threading.Thread(target=drive_sync_worker, daemon=True)\n",
        "    t.start()\n",
        "else:\n",
        "    print('[WARNING] No Google Drive checkpoint directory found. Background sync disabled.')\n",
    ]
    return make_code_cell(source)


def build_training_cell(meta: ModelNotebookMeta, target: str = "kaggle") -> dict[str, Any]:
    """Build process janitor and training coordinator launch code cell."""
    if target == "kaggle":
        suite_candidates = [
            "/kaggle/working/lemgendary-training-suite",
            "/kaggle/working/model-training/lemgendary-training-suite",
            "/kaggle/working",
        ]
        default_dir = "/kaggle/working/lemgendary-training-suite"
        env_flag = "kaggle"
    else:
        suite_candidates = [
            "/content/lemgendary-training-suite",
            "/content/model-training/lemgendary-training-suite",
            "/content",
        ]
        default_dir = "/content/lemgendary-training-suite"
        env_flag = "colab"

    launch_msg = "Forex Curriculum Orchestrator" if meta.is_forex else "Training Matrix"
    cmd_line = (
        "cmd = [sys.executable, '-u', '-m', 'training.train_forex_curriculum']\n"
        if meta.is_forex
        else f"cmd = [sys.executable, '-u', 'training/train.py', '--model', f'{{model_key}}', '--env', '{env_flag}', '--auto_sync']\n"
    )

    if target == "kaggle":
        exec_lines = [
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
            "    print('[OK] Subprocess successfully killed. VRAM and CPU are clean.')\n",
        ]
    else:
        exec_lines = [
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
            "    print('[OK] Subprocess successfully killed. VRAM and CPU are clean.')\n",
        ]

    source = [
        "import os, subprocess, sys\n",
        f"suite_candidates = {repr(suite_candidates)}\n",
        f"active_suite_dir = next((p for p in suite_candidates if os.path.exists(os.path.join(p, 'training', 'train.py'))), '{default_dir}')\n",
        "os.chdir(active_suite_dir)\n",
        "print(f'[OK] [SUITE] Active working directory set to: {os.getcwd()}')\n",
        "\n",
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
        f"print(f'[LAUNCH] [NUCLEAR] Initiating {launch_msg} for {meta.model_key}...')\n",
        cmd_line,
    ] + exec_lines
    return make_code_cell(source)


def build_push_cell(meta: ModelNotebookMeta, target: str = "kaggle") -> dict[str, Any]:
    """Build the KaggleHub and Google Drive deployment code cell."""
    local_path = (
        f"/kaggle/working/LemGendaryModels/{meta.model_key}"
        if target == "kaggle"
        else f"/content/LemGendaryModels/{meta.model_key}"
    )

    source = [
        "import os, kagglehub\n",
        f"model_key = '{meta.model_key}'\n",
        f"local_path = '{local_path}'\n",
        f"model_handle = '{meta.k_handle}'\n",
        "\n",
        "if os.path.exists(local_path):\n",
        "    print(f'[KAGGLE] Pushing finalized SOTA to {model_handle}...')\n",
        "    try:\n",
        f"        kagglehub.model_upload(model_handle, local_path, version_notes='v16.2.9 SOTA Finalized Sync: {meta.model_key}')\n",
        "        print('[DONE] Deployment Complete.')\n",
        "        print('[GDRIVE] Synchronizing finalized production artifacts to Google Drive...')\n",
        "        try:\n",
        "            from training.gdrive_cloud_manager import GDriveCloudManager\n",
        "            g_mgr = GDriveCloudManager(model_key)\n",
        "            g_mgr.sync()\n",
        "        except Exception as e:\n",
        "            print(f'[WARN] Google Drive final sync notice: {e}')\n",
        "    except Exception as e:\n",
        "        print(f'[ERROR] Deployment failed: {e}')\n",
        "        print('[REMEDY] Ensure your Kaggle API key is correctly configured and the destination kernel slug is valid.')\n",
        "else: print(f'[WARNING] Local manifold not found at {local_path}')\n",
    ]
    return make_code_cell(source)
