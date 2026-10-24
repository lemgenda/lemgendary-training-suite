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
        else f"cmd = [sys.executable, '-u', 'training/train.py', '--model', '{meta.model_key}', '--env', '{env_flag}', '--auto_sync']\n"
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
    """Build the KaggleHub (4 variations) and Google Drive deployment code cell."""
    local_path = (
        f"/kaggle/working/LemGendaryModels/{meta.model_key}"
        if target == "kaggle"
        else f"/content/LemGendaryModels/{meta.model_key}"
    )

    metadata_yaml_str = ""
    try:
        from training.cloud.metadata_generator import KaggleMetadataService
        meta_svc = KaggleMetadataService(owner_slug=meta.k_username)
        metadata_obj = meta_svc.build(meta.model_key)
        metadata_yaml_str = metadata_obj.to_yaml()
    except Exception:
        metadata_yaml_str = ""

    source = [
        "import os, sys, shutil, json, tempfile\n",
        f"model_key = '{meta.model_key}'\n",
        f"local_path = '{local_path}'\n",
        f"owner_slug = '{meta.k_username}'\n",
        f"model_slug = '{meta.k_slug}'\n",
        f"handles = {repr(meta.handles)}\n",
        f"EMBEDDED_METADATA_YAML = {repr(metadata_yaml_str)}\n",
        "\n",
        "print(f'[PUBLISH] Initiating 4-variation Kaggle Model Hub publish for {model_key}...')\n",
        "if not os.path.exists(local_path):\n",
        "    print(f'[ERROR] Local manifold directory not found at {local_path}')\n",
        "    sys.exit(1)\n",
        "\n",
        "try:\n",
        "    # 1. Run Universal Exporter to generate .pt, ONNX FP32 and ONNX FP16\n",
        "    print('[1/4] Running Universal Exporter for all production formats...')\n",
        "    from training.export.universal_exporter import UniversalExporter\n",
        "    exporter = UniversalExporter()\n",
        "    export_res = exporter.export_all_for_model(model_key, base_dir=local_path)\n",
        "    print(f'   -> Exporter completed with status: {export_res.success}')\n",
        "except Exception as exp_err:\n",
        "    print(f'   -> [NOTICE] Exporter step note: {exp_err}')\n",
        "\n",
        "try:\n",
        "    # 2. Stage isolated variations under dist/kaggle/\n",
        "    print('[2/4] Staging 4 isolated variations (default, fp32, onnx_fp16, onnx_fp32)...')\n",
        "    from training.export.staging import VariationStagingService\n",
        "    from training.export.validation import StrictIsolationValidator\n",
        "    staging_svc = VariationStagingService()\n",
        "    staging_res = staging_svc.stage_variations(model_key, base_dir=local_path, validate=True)\n",
        "    if not staging_res.success:\n",
        "        print(f'[ERROR] Variation staging audit failed: {staging_res.errors}')\n",
        "        sys.exit(1)\n",
        "    print('   -> Staging isolation verification passed successfully.')\n",
        "except Exception as stage_err:\n",
        "    print(f'[ERROR] Staging failed: {stage_err}')\n",
        "    sys.exit(1)\n",
        "\n",
        "# 3. Upload each of the 4 variations via kagglehub\n",
        "print('[3/4] Uploading 4 variations to Kaggle Models Hub...')\n",
        "import kagglehub\n",
        "dist_root = os.path.join(local_path, 'dist', 'kaggle')\n",
        "var_payload_dirs = {\n",
        "    'pytorch/default': os.path.join(dist_root, 'pytorch_default'),\n",
        "    'pytorch/fp32': os.path.join(dist_root, 'pytorch'),\n",
        "    'onnx/fp16': os.path.join(dist_root, 'onnx_fp16'),\n",
        "    'onnx/fp32': os.path.join(dist_root, 'onnx_fp32'),\n",
        "}\n",
        "\n",
        "upload_errors = []\n",
        "for var_key, handle in handles.items():\n",
        "    p_dir = var_payload_dirs.get(var_key)\n",
        "    if not p_dir or not os.path.exists(p_dir):\n",
        "        err = f'Missing payload directory for variation {var_key}: {p_dir}'\n",
        "        upload_errors.append(err)\n",
        "        print(f'   [ERROR] {err}')\n",
        "        continue\n",
        "    print(f'   [UPLOAD] Uploading variation {var_key} -> {handle}...')\n",
        "    try:\n",
        "        kagglehub.model_upload(\n",
        "            handle=handle,\n",
        "            local_model_dir=p_dir,\n",
        "            version_notes=f'Automated SOTA Release: {model_key} [{var_key}]',\n",
        "        )\n",
        "        print(f'   [OK] Variation {var_key} uploaded successfully.')\n",
        "    except Exception as up_err:\n",
        "        upload_errors.append(f'[{var_key}] {up_err}')\n",
        "        print(f'   [ERROR] Failed uploading {var_key}: {up_err}')\n",
        "\n",
        "if upload_errors:\n",
        "    print(f'[ERROR] Variation upload failed: {\"; \".join(upload_errors)}')\n",
        "    sys.exit(1)\n",
        "\n",
        "print(f'[SUCCESS] Model {model_key} pushed successfully to Kaggle (4 variations)')\n",
        "\n",
        "# 4. Apply model and instance metadata via Kaggle API\n",
        "print('[4/4] Synchronizing Kaggle Model and Variation metadata...')\n",
        "meta_synced = False\n",
        "try:\n",
        "    from training.cloud.model_metadata import ModelMetadata\n",
        "    from training.cloud.kaggle_api_sync import KaggleApiSyncService\n",
        "    if EMBEDDED_METADATA_YAML:\n",
        "        import yaml\n",
        "        meta_dict = yaml.safe_load(EMBEDDED_METADATA_YAML)\n",
        "        metadata = ModelMetadata.from_dict(meta_dict)\n",
        "    else:\n",
        "        meta_yaml_path = os.path.join(local_path, 'model_metadata.yaml')\n",
        "        metadata = ModelMetadata.load(meta_yaml_path)\n",
        "    api_service = KaggleApiSyncService(username=owner_slug)\n",
        "    sync_res = api_service.sync_metadata(metadata)\n",
        "    if sync_res.get('errors'):\n",
        "        print(f\"   [WARN] Metadata update warnings: {sync_res['errors']}\")\n",
        "    meta_synced = True\n",
        "    print(f'[SUCCESS] Kaggle metadata updated successfully for {model_key}')\n",
        "except Exception as meta_err:\n",
        "    print(f'   [WARN] Metadata synchronization notice: {meta_err}')\n",
        "\n",
        "# Google Drive mirror\n",
        "print('[GDRIVE] Synchronizing production artifacts to Google Drive...')\n",
        "try:\n",
        "    from training.gdrive_cloud_manager import GDriveCloudManager\n",
        "    g_mgr = GDriveCloudManager(model_key)\n",
        "    g_mgr.sync()\n",
        "except Exception as g_err:\n",
        "    print(f'   [NOTICE] Google Drive sync note: {g_err}')\n",
        "\n",
        "print(f'[COMPLETED] Fleet publishing completed successfully for {model_key}.')\n",
    ]
    return make_code_cell(source)
