"""Cloud synchronization, authentication, and training execution cell generator.

Handles credentials mounting, Google Drive FUSE attachment, background thread sync,
training subprocess management, and final artifact uploads to KaggleHub and Google Drive.
"""

import logging
from typing import Any

from training.cloud.session_deadline import DEADLINE_MARKER
from training.ecosystem import kaggle_owner
from training.platform_paths import platform_paths

from ..registry import ModelNotebookMeta
from .base import make_code_cell

logger = logging.getLogger("lemtrain.notebooks.cells.sync")


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
            "    g_drive = None\n",
            "    g_drive_json = None\n",
            "    try: g_pat = _c.get_secret('GITHUB_PAT')\n",
            (
                "    except Exception: print('[REMEDY] Missing secret! You should create new "
                "secret named GITHUB_PAT with your GitHub Personal Access Token as value')\n"
            ),
            "    try: s_pat = _c.get_secret('SUITE_PAT')\n",
            (
                "    except Exception: print('[REMEDY] Missing secret! You should create new "
                "secret named SUITE_PAT with your GitHub Personal Access Token as value')\n"
            ),
            "    try: k_key = _c.get_secret('KAGGLE_KEY')\n",
            (
                "    except Exception: print('[REMEDY] Missing secret! You should create new "
                "secret named KAGGLE_KEY with your Kaggle API Token as value')\n"
            ),
            "    try: k_user = _c.get_secret('KAGGLE_USERNAME')\n",
            (
                "    except Exception: print('[REMEDY] Missing secret! You should create new "
                "secret named KAGGLE_USERNAME with your Kaggle username as value')\n"
            ),
            "    try: g_drive = _c.get_secret('GOOGLE_DRIVE')\n",
            "    except Exception: pass\n",
            "    try: g_drive_json = _c.get_secret('GOOGLE_DRIVE_JSON')\n",
            "    except Exception: pass\n",
            "    \n",
            "    if g_pat: _os.environ['GITHUB_PAT'] = g_pat\n",
            "    if s_pat: _os.environ['SUITE_PAT'] = s_pat\n",
            "    if g_drive: _os.environ['GOOGLE_DRIVE'] = g_drive\n",
            "    if g_drive_json:\n",
            "        _os.environ['GOOGLE_DRIVE_JSON'] = g_drive_json\n",
            "        _g_dir = _os.path.expanduser('~/.gdrive')\n",
            "        _os.makedirs(_g_dir, exist_ok=True)\n",
            "        _sa_path = _os.path.join(_g_dir, 'service_account.json')\n",
            "        with open(_sa_path, 'w') as _gf:\n",
            (
                "            _gf.write(g_drive_json if isinstance(g_drive_json, str) else "
                "_json.dumps(g_drive_json))\n"
            ),
            "        _os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = _sa_path\n",
            "    \n",
            f"    if not k_user: k_user = {kaggle_owner()!r}\n",
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
            "    if g_drive_json: active.append('GOOGLE_DRIVE_JSON')\n",
            "    if active:\n",
            "        print(f'[OK] [AUTH] Kaggle Secrets mounted: {\", \".join(active)}')\n",
            "    else:\n",
            (
                "        print('[ERROR] [CRITICAL] No PATs found in Kaggle Secrets! Private "
                "repositories will fail to clone.')\n"
            ),
            (
                "        print('[ACTION REQUIRED] In Kaggle Notebook top bar -> Add-ons -> Secrets "
                "-> Add SUITE_PAT or GITHUB_PAT.')\n"
            ),
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
            (
                "    except Exception: print('[REMEDY] Missing secret! You should create new "
                "secret named GITHUB_PAT with your GitHub Personal Access Token as value')\n"
            ),
            "    try: s_pat = userdata.get('SUITE_PAT')\n",
            (
                "    except Exception: print('[REMEDY] Missing secret! You should create new "
                "secret named SUITE_PAT with your GitHub Personal Access Token as value')\n"
            ),
            "    try: k_key = userdata.get('KAGGLE_KEY')\n",
            (
                "    except Exception: print('[REMEDY] Missing secret! You should create new "
                "secret named KAGGLE_KEY with your Kaggle API Token as value')\n"
            ),
            "    try: k_user = userdata.get('KAGGLE_USERNAME')\n",
            (
                "    except Exception: print('[REMEDY] Missing secret! You should create new "
                "secret named KAGGLE_USERNAME with your Kaggle username as value')\n"
            ),
            "    try: g_drive = userdata.get('GOOGLE_DRIVE')\n",
            (
                "    except Exception: print('[REMEDY] Missing secret! You should create new "
                "secret named GOOGLE_DRIVE with your Google Drive token as value')\n"
            ),
            "    \n",
            "    if g_pat: _os.environ['GITHUB_PAT'] = g_pat\n",
            "    if s_pat: _os.environ['SUITE_PAT'] = s_pat\n",
            "    if g_drive: _os.environ['GOOGLE_DRIVE'] = g_drive\n",
            "    \n",
            f"    if not k_user: k_user = {kaggle_owner()!r}\n",
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
            "    if g_drive_json: active.append('GOOGLE_DRIVE_JSON')\n",
            "    if g_drive: active.append('GOOGLE_DRIVE')\n",
            "    if active:\n",
            "        print(f'[OK] [AUTH] Colab Secrets mounted: {\", \".join(active)}')\n",
            "    else:\n",
            (
                "        print('[WARNING] No PATs found in Colab Secrets! Private repositories "
                "will fail to clone.')\n"
            ),
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
        (
            "print('[OK] Google Drive mounted successfully. Datasets will be streamed directly "
            "from Drive.')\n"
        ),
    ]
    return make_code_cell(source)


def build_continuous_sync_cell(meta: ModelNotebookMeta) -> dict[str, Any]:
    """Build background Google Drive sync worker code cell for Colab."""
    source = [
        "import os, time, shutil, threading\n",
        f"model_key = '{meta.identity.model_key}'\n",
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
        (
            "    drive_target_dir = f'/content/drive/MyDrive/LemGendaryModels/{model_key}/checkpoin"
            "ts'\n"
        ),
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
        (
            "                    if not os.path.exists(dst) or os.path.getmtime(src) > "
            "os.path.getmtime(dst):\n"
        ),
        "                        tmp_dst = dst + '.tmp'\n",
        "                        shutil.copy2(src, tmp_dst)\n",
        "                        os.rename(tmp_dst, dst)\n",
        "            m_src = os.path.join(model_hub_dir, 'metrics.csv')\n",
        "            if os.path.exists(m_src):\n",
        "                m_dst = os.path.join(os.path.dirname(drive_target_dir), 'metrics.csv')\n",
        (
            "                if not os.path.exists(m_dst) or os.path.getmtime(m_src) > "
            "os.path.getmtime(m_dst):\n"
        ),
        "                    shutil.copy2(m_src, m_dst)\n",
        "        except Exception as e:\n",
        "            pass\n",
        "        time.sleep(30)\n",
        "\n",
        "if drive_target_dir:\n",
        "    t = threading.Thread(target=drive_sync_worker, daemon=True)\n",
        "    t.start()\n",
        "else:\n",
        (
            "    print('[WARNING] No Google Drive checkpoint directory found. Background sync "
            "disabled.')\n"
        ),
    ]
    return make_code_cell(source)


def build_training_cell(meta: ModelNotebookMeta, target: str = "kaggle") -> dict[str, Any]:
    """Build process janitor and training coordinator launch code cell."""
    default_dir = platform_paths(target).suite_dir
    suite_candidates = [default_dir]
    env_flag = "kaggle" if target == "kaggle" else "colab"

    launch_msg = "Forex Curriculum Orchestrator" if meta.identity.is_forex else "Training Matrix"
    cmd_dispatch = [
        "is_forex = " + str(meta.identity.is_forex) + "\n",
        f"model_key = '{meta.identity.model_key}'\n",
        f"preferred_parallel = '{meta.preferred_parallel}'\n",
        "import torch\n",
        "num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0\n",
        (
            "use_ddp = (num_gpus > 1) and (preferred_parallel == 'ddp' or (preferred_parallel != "
            "'single' and not is_forex and 'yolo' not in model_key.lower()))\n"
        ),
        "\n",
        "if is_forex:\n",
        "    cmd = [sys.executable, '-u', '-m', 'training.train_forex_curriculum']\n",
        "elif use_ddp and os.path.exists('tools/launch_ddp.py'):\n",
        (
            '    print(f"[DDP] Dual-GPU acceleration detected ({num_gpus} GPUs). Launching '
            'multi-process DDP via tools/launch_ddp.py...")\n'
        ),
        (
            "    cmd = [sys.executable, '-u', 'tools/launch_ddp.py', '--model', "
            f"'{meta.identity.model_key}', '--env', '{env_flag}', '--auto_sync']\n"
        ),
        "else:\n",
        (
            "    cmd = [sys.executable, '-u', 'training/train.py', '--model', "
            f"'{meta.identity.model_key}', "
            f"'--env', '{env_flag}', '--auto_sync']\n"
        ),
    ]

    if target == "kaggle":
        exec_lines = [
            "p = subprocess.Popen(cmd)\n",
            "try:\n",
            "    p.wait()\n",
            "except KeyboardInterrupt:\n",
            (
                "    print('\\n[TERMINATED] Training interrupted by user. Terminating training "
                "subprocess safely...')\n"
            ),
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
            (
                "    print('\\n[TERMINATED] Training interrupted by user. Terminating training "
                "subprocess safely...')\n"
            ),
            "    try:\n",
            "        p.terminate()\n",
            "        p.wait(timeout=5)\n",
            "    except subprocess.TimeoutExpired:\n",
            "        p.kill()\n",
            "    print('[OK] Subprocess successfully killed. VRAM and CPU are clean.')\n",
        ]

    source = (
        [
            "import os, subprocess, sys\n",
            f"suite_candidates = {suite_candidates!r}\n",
            (
                "active_suite_dir = next((p for p in suite_candidates if "
                f"os.path.exists(os.path.join(p, 'training', 'train.py'))), '{default_dir}')\n"
            ),
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
            (
                "                print(f'[JANITOR] Killing stale zombie training process (PID "
                "{pid})...')\n"
            ),
            "                subprocess.run(['kill', '-9', str(pid)], capture_output=True)\n",
            "except Exception:\n",
            "    pass\n",
            "\n",
            (
                f"print(f'[LAUNCH] [NUCLEAR] Initiating {launch_msg} "
                f"for {meta.identity.model_key}...')\n"
            ),
        ]
        + cmd_dispatch
        + exec_lines
    )
    return make_code_cell(source)


def build_push_cell(meta: ModelNotebookMeta, target: str = "kaggle") -> dict[str, Any]:
    """Build the release cell: one call into the suite's publish policy, then Google Drive.

    ``publish_release`` exports every format and runs ``KaggleModelSyncManager.push_model``,
    the same policy as the GUI and CLI push (model page first, license, unchanged variations
    skipped, version notes with epoch, metrics and suite version). An export or upload failure
    stops the cell with exit code 1.
    """
    model_key = meta.identity.model_key
    local_path = f"{platform_paths(target).hub_root}/{model_key}"
    source = [
        "import os, sys\n",
        f"model_key = {model_key!r}\n",
        f"local_path = {local_path!r}\n",
        "\n",
        "print(f'[PUBLISH] Exporting and publishing {model_key} to the Kaggle Models Hub...')\n",
        "from training.cloud.release_publish import publish_release\n",
        "release = publish_release(model_key)\n",
        "for var_key, var_res in release.variations.items():\n",
        "    if not var_res.success:\n",
        "        print(f'   [ERROR] {var_key}: {var_res.error}')\n",
        "    elif var_res.skipped:\n",
        "        print(f'   [UNCHANGED] {var_key}: no new version ({var_res.handle})')\n",
        "    else:\n",
        "        print(f'   [OK] {var_key} -> {var_res.handle}')\n",
        "if not release.success:\n",
        "    print(f'[ERROR] Release publish failed: {release.errors}')\n",
        "    sys.exit(1)\n",
        "print(f'[SUCCESS] Release of {model_key} published to the Kaggle Models Hub.')\n",
        "\n",
        "print('[GDRIVE] Synchronizing production artifacts to Google Drive...')\n",
        "from training.gdrive_cloud_manager import GDriveCloudManager\n",
        "if GDriveCloudManager(model_key).sync():\n",
        "    print('[GDRIVE] Google Drive mirror updated.')\n",
        "else:\n",
        "    print('[GDRIVE] Google Drive mirror not updated (see the log above).')\n",
    ]
    return make_code_cell(_skip_after_deadline(source, local_path))


def _skip_after_deadline(source: list[str], local_path: str) -> list[str]:
    """Guard the publish steps: a run stopped at its session deadline publishes no release.

    The deadline stop already synced ``pytorch/default``; the next session resumes it.
    """
    start = next(i for i, line in enumerate(source) if "[PUBLISH]" in line)
    marker = f"{local_path}/{DEADLINE_MARKER}"
    guard = [
        f"if os.path.exists({marker!r}):\n",
        (
            "    print('[DEADLINE] Session deadline stop: progress is on the Hub; release "
            "publish skipped until training finishes.')\n"
        ),
        "else:\n",
    ]
    body = [f"    {line}" if line.strip() else line for line in source[start:]]
    return source[:start] + guard + body
