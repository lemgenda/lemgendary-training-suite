"""Model hub preparation, weight recovery, and checkpoint hydration cell generator.

Initializes model manifolds in LemGendaryModels and searches KaggleHub / Google Drive
for prior checkpoints and metrics histories.
"""

from typing import Any

from training.platform_paths import platform_paths

from ..registry import ModelNotebookMeta
from .base import make_code_cell


def build_hub_prep_cell(meta: ModelNotebookMeta, target: str = "kaggle") -> dict[str, Any]:
    """Build the model hub structure initialization code cell."""
    hub_root = platform_paths(target).hub_root
    source = [
        "import os\n",
        f"hub_root = '{hub_root}'\n",
        f"model_key = '{meta.identity.model_key}'\n",
        "model_dir = os.path.join(hub_root, model_key)\n",
        "ckpt_dir = os.path.join(model_dir, 'checkpoints')\n",
        "nb_dir = os.path.join(model_dir, 'notebooks')\n",
        "vis_dir = os.path.join(model_dir, 'visualizations')\n",
        "dist_dir = os.path.join(model_dir, 'dist', 'kaggle')\n",
        "\n",
        "print(f'[HUB] Initializing Canonical Manifold for {model_key}...')\n",
        "os.makedirs(ckpt_dir, exist_ok=True)\n",
        "os.makedirs(nb_dir, exist_ok=True)\n",
        "os.makedirs(vis_dir, exist_ok=True)\n",
        "os.makedirs(os.path.join(dist_dir, 'pytorch'), exist_ok=True)\n",
        "os.makedirs(os.path.join(dist_dir, 'onnx_fp32'), exist_ok=True)\n",
        "os.makedirs(os.path.join(dist_dir, 'onnx_fp16'), exist_ok=True)\n",
        "print(f'[OK] Canonical manifold structure ready at {model_dir}')\n",
    ]
    return make_code_cell(source)


def _suite_import_lines(target: str) -> list[str]:
    """Put the suite checkout on ``sys.path`` and import the one model-file loader."""
    return [
        "import glob, os, shutil, sys\n",
        f"suite_path = {platform_paths(target).suite_dir!r}\n",
        "if suite_path not in sys.path:\n",
        "    sys.path.insert(0, suite_path)\n",
        "from training.utils.model_payload import checkpoint_epoch, load_model_payload\n",
    ]


def _model_dir_lines(meta: ModelNotebookMeta, target: str) -> list[str]:
    paths = platform_paths(target)
    return [
        f"model_key = {meta.identity.model_key!r}\n",
        f"model_slug = {meta.kaggle.slug!r}\n",
        f"model_hub_dir = {paths.hub_root + '/' + meta.identity.model_key!r}\n",
        "ckpt_hub_dir = os.path.join(model_hub_dir, 'checkpoints')\n",
        "os.makedirs(ckpt_hub_dir, exist_ok=True)\n",
    ]


_RECOVERY_HELPERS = [
    "\n",
    "def _target_name(fname):\n",
    "    # latest/last, best, progress and vault_* checkpoints; anything else is not resumed.\n",
    "    stem, ext = os.path.splitext(fname)\n",
    "    low = stem.lower()\n",
    "    if ext.lower() not in ('.pth', '.pt'):\n",
    "        return None\n",
    "    if low.startswith('vault_'):\n",
    "        return fname\n",
    "    for key, name in (('progress', 'progress'), ('best', 'best'), ('latest', 'latest'),\n",
    "                      ('last', 'latest')):\n",
    "        if key in low:\n",
    "            return name + ext.lower()\n",
    "    return None\n",
    "\n",
    "def _model_files(root):\n",
    "    found = []\n",
    "    for dirpath, _, files in os.walk(root):\n",
    "        found += [os.path.join(dirpath, f) for f in files if _target_name(f)]\n",
    "    return sorted(found)\n",
    "\n",
    "def _model_dirs(root, max_depth=7):\n",
    "    # Folders named after the Kaggle slug (hyphenated) or the model key.\n",
    "    names = {model_slug.lower(), model_key.lower()}\n",
    "    hits, queue = [], [(root, 0)]\n",
    "    while queue:\n",
    "        curr, depth = queue.pop(0)\n",
    "        try:\n",
    "            entries = sorted(os.scandir(curr), key=lambda e: e.name)\n",
    "        except OSError as scan_err:\n",
    "            print(f'   -> [SKIP] {curr}: {scan_err}')\n",
    "            continue\n",
    "        for entry in entries:\n",
    "            if not entry.is_dir():\n",
    "                continue\n",
    "            if entry.name.lower() in names:\n",
    "                hits.append(entry.path)\n",
    "            elif depth < max_depth:\n",
    "                queue.append((entry.path, depth + 1))\n",
    "    return hits\n",
    "\n",
    "adopted = {}\n",
    "def _adopt(src, label):\n",
    "    # Keep the checkpoint with the newest stored epoch; never compare file sizes.\n",
    "    name = _target_name(os.path.basename(src))\n",
    "    dst = os.path.join(ckpt_hub_dir, name)\n",
    "    src_epoch = checkpoint_epoch(src)\n",
    "    if src_epoch is None:\n",
    "        print(f'   -> [SKIP] {src}: no readable epoch')\n",
    "        return\n",
    "    dst_epoch = checkpoint_epoch(dst) if os.path.exists(dst) else None\n",
    "    if dst_epoch is not None and src_epoch <= dst_epoch:\n",
    "        return\n",
    "    shutil.copy2(src, dst)\n",
    "    adopted[name] = (src_epoch, os.path.dirname(src))\n",
    "    print(f'   -> [OK] {label}: {os.path.basename(src)} (epoch {src_epoch}) -> {name}')\n",
    "\n",
]

_METRICS_LINES = [
    "\n",
    "latest = adopted.get('latest.pth') or adopted.get('latest.pt')\n",
    "if latest:\n",
    "    src_dir = latest[1]\n",
    "    for d in (os.path.dirname(src_dir), src_dir):\n",
    "        m_path = os.path.join(d, 'metrics.csv')\n",
    "        if os.path.exists(m_path):\n",
    "            shutil.copy2(m_path, os.path.join(model_hub_dir, 'metrics.csv'))\n",
    "            print(f'[METRICS] metrics.csv taken with the epoch {latest[0]} checkpoint')\n",
    "            break\n",
    "if not adopted:\n",
    "    print('   -> [SKIP] No newer checkpoint found; keeping the local hub as it is.')\n",
]


def build_checkpoint_recovery_cell(
    meta: ModelNotebookMeta, target: str = "kaggle"
) -> dict[str, Any]:
    """Build the checkpoint and metrics recovery code cell.

    Kaggle: the latest ``pytorch/default`` Hub version is loaded first (every model file in
    the explicit download folder, no name filter); ``/kaggle/input`` folders named after the
    Kaggle slug or the model key are used only when their checkpoints are newer. Colab: Google
    Drive folders named after the slug or key. Every choice compares the epoch stored in the
    checkpoint (``training.utils.model_payload.checkpoint_epoch``), never the file size, so a
    pinned old ``model_sources`` version cannot replace newer weights.
    """
    source = [
        *_suite_import_lines(target),
        *_model_dir_lines(meta, target),
        "print(f'[RECOVERY] Looking for {model_key} checkpoints ({model_slug})...')\n",
        *_RECOVERY_HELPERS,
    ]
    if target == "kaggle":
        source += [
            "import kagglehub\n",
            "from kagglehub import exceptions as kh_exc\n",
            "hub_errors = (OSError, ValueError, kh_exc.BackendError, kh_exc.NotFoundError,\n",
            "              kh_exc.CredentialError, kh_exc.UnauthenticatedError,\n",
            "              kh_exc.DataCorruptionError, kh_exc.KaggleEnvironmentError)\n",
            "try:\n",
            f"    hub_dir = kagglehub.model_download({meta.kaggle.pytorch_default!r})\n",
            "except hub_errors as kh_err:\n",
            "    hub_dir = None\n",
            "    print(f'   -> [KAGGLEHUB] No Hub version: {kh_err}')\n",
            "if hub_dir and os.path.isdir(hub_dir):\n",
            "    for src in _model_files(hub_dir):\n",
            "        _adopt(src, 'Hub latest')\n",
            "if os.path.isdir('/kaggle/input'):\n",
            "    for mount in _model_dirs('/kaggle/input'):\n",
            "        for src in _model_files(mount):\n",
            "            _adopt(src, 'Input mount (newer)')\n",
        ]
    else:
        source += [
            "drive_root = '/content/drive/MyDrive'\n",
            "if os.path.isdir(drive_root):\n",
            "    for folder in _model_dirs(drive_root):\n",
            "        for src in _model_files(folder):\n",
            "            _adopt(src, 'Google Drive')\n",
        ]
    source += _METRICS_LINES
    return make_code_cell(source)


def build_stealth_cell(meta: ModelNotebookMeta, target: str = "kaggle") -> dict[str, Any]:
    """Build the cell that loads the newest local checkpoint (by stored epoch).

    Every ``.pth`` and ``.pt`` file of the hub ``checkpoints`` folder is considered; the one
    with the highest stored epoch is loaded through the shared model-file loader.
    """
    source = [
        *_suite_import_lines(target),
        "import torch\n",
        *_model_dir_lines(meta, target),
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "ckpts = [p for ext in ('*.pth', '*.pt')\n",
        "         for p in glob.glob(os.path.join(ckpt_hub_dir, ext))]\n",
        "scored = sorted((e, p) for p in ckpts if (e := checkpoint_epoch(p)) is not None)\n",
        "if scored:\n",
        "    epoch, model_path = scored[-1]\n",
        "    payload = load_model_payload(model_path, map_location=device, trust_pickle=True)\n",
        "    print(f'[SOTA] Loaded {os.path.basename(model_path)} (epoch {epoch}, '\n",
        "          f'{len(payload.state_dict)} tensors) on {device}.')\n",
        "else:\n",
        "    print('[WARNING] No existing weights found. Starting from scratch.')\n",
    ]
    return make_code_cell(source)
