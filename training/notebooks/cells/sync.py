"""Cloud synchronization, authentication, and training execution cell generator.

Handles credentials mounting, Google Drive FUSE attachment, background thread sync,
training subprocess management, and final artifact uploads to KaggleHub and Google Drive.
"""

import logging
from typing import Any

from training.cloud.session_deadline import DEADLINE_MARKER
from training.platform_paths import platform_paths

from ..launch_options import DEFAULT_LAUNCH, LaunchOptions
from ..registry import ModelNotebookMeta
from .base import make_code_cell

logger = logging.getLogger("lemtrain.notebooks.cells.sync")


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
        "    last_error = None\n",
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
        "            last_error = None\n",
        "        except OSError as sync_err:\n",
        "            # Drive unmounted, quota full or a checkpoint mid-write (shutil.Error is an\n",
        "            # OSError): report each new error once, retry on the next pass.\n",
        "            if str(sync_err) != last_error:\n",
        "                last_error = str(sync_err)\n",
        "                print(f'[SYNC] [WARNING] Drive sync pass failed, retrying in 30 s: '\n",
        "                      f'{type(sync_err).__name__}: {sync_err}')\n",
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


_INTERRUPT_LINES = [
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


_GPU_COUNT_LINES = [
    "# GPUs are counted in a fresh interpreter: torch in this kernel process may be older\n",
    "# than the build the dependency cell installed.\n",
    "gpu_probe = subprocess.run(\n",
    "    [sys.executable, '-c',\n",
    "     'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)'],\n",
    "    capture_output=True, text=True)\n",
    "if gpu_probe.returncode == 0 and gpu_probe.stdout.strip():\n",
    "    num_gpus = int(gpu_probe.stdout.split()[-1])\n",
    "else:\n",
    "    num_gpus = 0\n",
    "    print(f'[WARNING] GPU count probe failed with exit code {gpu_probe.returncode}: '\n",
    "          f'{gpu_probe.stderr[-1000:]}')\n",
]


def _exit_gate_lines(meta: ModelNotebookMeta, target: str) -> list[str]:
    """Stop the notebook unless the trainer exited 0 (``training.cloud.trainer_exit``).

    A non-zero exit (failure, interruption, resume refusal) raises ``TrainerExitError``, so
    the kernel ends with an error and the publish cells never run. Exit 0 with the session
    deadline marker is the intentional deadline stop the next session resumes.
    """
    model_dir = f"{platform_paths(target).hub_root}/{meta.identity.model_key}"
    return [
        "\n",
        "if active_suite_dir not in sys.path:\n",
        "    sys.path.insert(0, active_suite_dir)\n",
        "from pathlib import Path\n",
        "from training.cloud.trainer_exit import TrainerOutcome, check_trainer_exit\n",
        f"trainer_outcome = check_trainer_exit(p.wait(), Path({model_dir!r}))\n",
        "if trainer_outcome is TrainerOutcome.DEADLINE_STOP:\n",
        (
            "    print('[DEADLINE] Training stopped at the session deadline; progress is "
            "synced and the next session resumes it.')\n"
        ),
        "else:\n",
        "    print('[OK] Training finished with exit code 0.')\n",
    ]


def _env_text(env: dict[str, str]) -> str:
    """``NAME=value`` pairs of ``env`` for a launch log line."""
    return " ".join(f"{name}={value}" for name, value in env.items())


def build_training_cell(
    meta: ModelNotebookMeta, target: str = "kaggle", options: LaunchOptions = DEFAULT_LAUNCH
) -> dict[str, Any]:
    """Build process janitor and training coordinator launch code cell.

    ``options`` add trainer switches (``--override-frozen``) and the session window override
    (environment variables the trainer's session deadline reads) to every launch path. The
    cell ends with the trainer exit gate (:func:`_exit_gate_lines`).
    """
    default_dir = platform_paths(target).suite_dir
    suite_candidates = [default_dir]
    env_flag = "kaggle" if target == "kaggle" else "colab"

    launch_msg = "Forex Curriculum Orchestrator" if meta.identity.is_forex else "Training Matrix"
    cmd_dispatch = [
        "is_forex = " + str(meta.identity.is_forex) + "\n",
        f"model_key = '{meta.identity.model_key}'\n",
        f"preferred_parallel = '{meta.preferred_parallel}'\n",
        *_GPU_COUNT_LINES,
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
    flags = options.trainer_flags()
    if flags:
        cmd_dispatch += [
            f"cmd += {flags!r}\n",
            f"print('[LAUNCH] Trainer switches: {' '.join(flags)}')\n",
        ]
    session_env = options.trainer_environment()
    if session_env:
        cmd_dispatch += [
            f"os.environ.update({session_env!r})\n",
            f"print({'[LAUNCH] Session window override: ' + _env_text(session_env)!r})\n",
        ]

    if target == "kaggle":
        run_lines = [
            "p = subprocess.Popen(cmd)\n",
            "try:\n",
            "    p.wait()\n",
        ]
    else:
        run_lines = [
            "p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n",
            "try:\n",
            "    import io\n",
            "    with io.TextIOWrapper(p.stdout, newline='') as trainer_out:\n",
            "        for line in trainer_out:\n",
            "            print(line, end='', flush=True)\n",
            "    p.wait()\n",
        ]
    exec_lines = run_lines + _INTERRUPT_LINES + _exit_gate_lines(meta, target)

    source = (
        [
            "import os, subprocess, sys\n",
            f"suite_candidates = {suite_candidates!r}\n",
            (
                "active_suite_dir = next((p for p in suite_candidates if "
                f"os.path.exists(os.path.join(p, 'training', 'train.py'))), {default_dir!r})\n"
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
            "except (OSError, ValueError, subprocess.SubprocessError) as janitor_err:\n",
            "    print(f'[JANITOR] Stale process scan skipped: {janitor_err}')\n",
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


_RUN_SUITE_LINES = [
    "\n",
    "def _run_suite(module, *args):\n",
    "    # Suite code that may load torch runs in a fresh interpreter, never in this kernel\n",
    "    # process: its torch may be older than the build the dependency cell installed.\n",
    "    with subprocess.Popen([sys.executable, '-u', '-m', module, *args], cwd=suite_dir,\n",
    "                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n",
    "                          text=True) as proc:\n",
    "        for line in proc.stdout:\n",
    "            print(line, end='', flush=True)\n",
    "    return proc.returncode\n",
]
"""Generated helper of the publish cell: run ``python -m <module>`` from the suite checkout,
stream its output into the cell and return its exit code."""


def build_push_cell(meta: ModelNotebookMeta, target: str = "kaggle") -> dict[str, Any]:
    """Build the release cell: one call into the suite's publish policy, then Google Drive.

    ``training.cloud.release_publish_cli`` runs ``publish_release``, which exports every format
    and runs ``KaggleModelSyncManager.push_model``, the same policy as the GUI and CLI push
    (model page first, license, unchanged variations skipped, version notes with epoch, metrics
    and suite version). Both steps run in a fresh interpreter (:data:`_RUN_SUITE_LINES`), never
    in the notebook's own kernel process. A failed export or upload raises ``RuntimeError``,
    so the kernel ends with an error; a failed Google Drive mirror is reported only.
    """
    model_key = meta.identity.model_key
    paths = platform_paths(target)
    local_path = f"{paths.hub_root}/{model_key}"
    source = [
        "import os, subprocess, sys\n",
        f"model_key = {model_key!r}\n",
        f"local_path = {local_path!r}\n",
        f"suite_dir = {paths.suite_dir!r}\n",
        *_RUN_SUITE_LINES,
        "\n",
        "print(f'[PUBLISH] Exporting and publishing {model_key} to the Kaggle Models Hub...')\n",
        "publish_code = _run_suite('training.cloud.release_publish_cli', model_key)\n",
        "if publish_code != 0:\n",
        (
            "    raise RuntimeError(f'[ERROR] Release publish of {model_key} failed with exit "
            "code {publish_code}; see the log above.')\n"
        ),
        "\n",
        "print('[GDRIVE] Synchronizing production artifacts to Google Drive...')\n",
        "gdrive_code = _run_suite('training.gdrive_cloud_manager', '--model', model_key)\n",
        "if gdrive_code == 0:\n",
        "    print('[GDRIVE] Google Drive mirror updated.')\n",
        "else:\n",
        (
            "    print(f'[GDRIVE] Google Drive mirror not updated (exit code {gdrive_code}; see "
            "the log above).')\n"
        ),
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
