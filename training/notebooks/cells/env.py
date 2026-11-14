"""Runtime environment and hardware sentinel cell generator.

Embeds runtime environment configuration from LemGendary Environment Manager
and generates hardware probe sentinel cells.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from training.cloud.session_deadline import SESSION_START_ENV

from .base import make_code_cell

_RUNTIME_ENV_FALLBACK: dict[str, str] = {
    "PYTHONUTF8": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
}

_BLOCKED_RUNTIME_ENV: set[str] = {
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_FORCE_PTX_JIT",
    "TORCH_CUDA_ARCH_LIST",
    "CUDA_CACHE_PATH",
    "CUDA_CACHE_MAXSIZE",
}


def _load_runtime_env() -> dict[str, str]:
    """Read runtime_env.yaml from env-manager if present, else return defaults.

    Tries several candidate paths so this works regardless of how deeply the
    generator is nested relative to the env-manager repo.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent.parent.parent
        / "lemgendary-env-manager"
        / "requirements"
        / "runtime_env.yaml",
        here.parent.parent.parent.parent
        / "lemgendary-env-manager"
        / "requirements"
        / "runtime_env.yaml",
        here.parent.parent.parent / "lemgendary-env-manager" / "requirements" / "runtime_env.yaml",
        here.parent.parent / "lemgendary-env-manager" / "requirements" / "runtime_env.yaml",
    ]
    yaml_path = next((p for p in candidates if p.exists()), None)
    if yaml_path is None:
        return dict(_RUNTIME_ENV_FALLBACK)

    try:
        raw = yaml_path.read_text(encoding="utf-8")
    except OSError:
        return dict(_RUNTIME_ENV_FALLBACK)

    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return dict(_RUNTIME_ENV_FALLBACK)

    result: dict[str, str] = {}
    for name, spec in (data.get("variables") or {}).items():
        if isinstance(spec, dict) and "value" in spec:
            result[str(name)] = str(spec["value"])

    result = {k: v for k, v in result.items() if k not in _BLOCKED_RUNTIME_ENV}
    # Default standard allocator

    return result if result else dict(_RUNTIME_ENV_FALLBACK)


def _build_env_var_lines(env_vars: dict[str, str]) -> list[str]:
    """Return Python source lines that assign each env var, sorted by name."""
    lines: list[str] = []
    for name in sorted(env_vars.keys()):
        value = env_vars[name].replace("\\", "\\\\").replace("'", "\\'")
        lines.append(f"os.environ['{name}'] = '{value}'\n")
    return lines


NO_GPU_EXIT_CODE = 3
"""Exit code of the hardware probe when no CUDA GPU is visible."""


@dataclass(frozen=True)
class _SentinelText:
    """Platform wording of the hardware sentinel."""

    recommended: str
    incompatible_remedy: str
    no_gpu_remedy: str
    abort: str


_SENTINEL_TEXT = {
    "kaggle": _SentinelText(
        recommended="Kaggle: GPU T4 x2 (30GB total VRAM) [Recommended]",
        incompatible_remedy=(
            "Switch Kaggle Accelerator to GPU T4 x2 in Notebook Settings (right panel)."
        ),
        no_gpu_remedy=(
            "   -> Kaggle: Right Panel -> Session Options -> Accelerator -> "
            "GPU T4 x2 (30GB total VRAM) [Recommended]"
        ),
        abort="Enable GPU in Session Options and re-run from the top.",
    ),
    "colab": _SentinelText(
        recommended="Google Colab: T4 GPU (or A100/L4 with Pro)",
        incompatible_remedy=(
            "Switch Colab Runtime to T4 GPU (Runtime -> Change runtime type -> T4 GPU)."
        ),
        no_gpu_remedy=(
            "   -> Colab:  Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU"
        ),
        abort="Enable GPU in Colab Runtime settings and re-run from the top.",
    ),
}


def _hardware_probe_lines(text: _SentinelText) -> list[str]:
    """Source of the torch hardware probe the sentinel runs in a fresh interpreter.

    It exits :data:`NO_GPU_EXIT_CODE` when no CUDA GPU is visible. The lines contain no
    backslash and no triple quote, so the cell embeds them in a triple-quoted string.
    """
    return [
        "import os, sys\n",
        "import torch\n",
        "tpu_detected = bool(\n"
        "    os.environ.get('TPU_NAME') or os.environ.get('KAGGLE_TPU_NAME')\n"
        "    or os.path.exists('/dev/vfio/0')\n"
        ")\n",
        "xla_devices = []\n",
        "if not torch.cuda.is_available():\n",
        "    try:\n",
        "        os.environ.setdefault('PJRT_DEVICE', 'TPU')\n",
        "        import torch_xla.core.xla_model as xm\n",
        "        xla_devices = xm.get_xla_supported_devices()\n",
        "        tpu_detected = bool(xla_devices)\n",
        "    except Exception:\n",
        "        pass\n",
        "if xla_devices:\n",
        "    print(f'[OK] [HARDWARE] Google Cloud TPU XLA validated '\n",
        "          f'({len(xla_devices)} cores).')\n",
        "    sys.exit(0)\n",
        "if tpu_detected:\n",
        "    print('[OK] [HARDWARE] TPU VM detected; torch_xla will be initialized in deps.')\n",
        "    sys.exit(0)\n",
        "if not torch.cuda.is_available():\n",
        "    print('[CRITICAL ERROR] [HARDWARE] NO GPU OR TPU DETECTED! '\n",
        "          'Training cannot proceed on CPU.')\n",
        "    print('[ACTION REQUIRED] Enable GPU or TPU Accelerator before '\n",
        "          'running this notebook:')\n",
        f"    print({text.no_gpu_remedy!r})\n",
        f"    sys.exit({NO_GPU_EXIT_CODE})\n",
        "gpu_name = torch.cuda.get_device_name(0)\n",
        "cap = torch.cuda.get_device_capability(0)\n",
        "archs = torch.cuda.get_arch_list()\n",
        "compat = any(f'{cap[0]}.{cap[1]}' in a or f'sm_{cap[0]}{cap[1]}' in a for a in archs)\n",
        "try:\n",
        "    probe = torch.ones(1, device='cuda') + 1.0\n",
        "    torch.cuda.synchronize()\n",
        "    del probe\n",
        "except RuntimeError as kernel_err:\n",
        "    print(f'[WARNING] [HARDWARE] CUDA kernel test failed: {kernel_err}')\n",
        "    compat = False\n",
        "if compat:\n",
        "    print(f'[OK] [HARDWARE] NVIDIA {gpu_name} (sm_{cap[0]}{cap[1]}) '\n",
        "          'validated & ready.')\n",
        "else:\n",
        "    print(f'[CRITICAL ERROR] [HARDWARE] NVIDIA {gpu_name} '\n",
        "          f'(sm_{cap[0]}{cap[1]}) has no '\n",
        "          'kernel images in the preinstalled PyTorch build.')\n",
        f"    print({'[ACTION REQUIRED] ' + text.incompatible_remedy!r})\n",
        "    print('[INFO] The dependency cell installs the suite torch build for this GPU.')\n",
        "props = torch.cuda.get_device_properties(0)\n",
        "tag = '[OK] [ACTIVE]' if compat else '[WARNING] [INCOMPATIBLE ACCELERATOR]'\n",
        "print(f'{tag} {props.name} (Compute Capability sm_{cap[0]}{cap[1]})')\n",
        "print(f'[OK] [VRAM] {props.total_memory / 1024**3:.1f} GB')\n",
        "if torch.cuda.device_count() > 1:\n",
        "    print(f'[OK] [MULTI-GPU] Detected {torch.cuda.device_count()} active GPUs.')\n",
        "if props.total_memory / 1024**3 < 10.0:\n",
        "    print('[WARNING] Low VRAM detected. '\n",
        "          'Suite will enable Survival Profiles automatically.')\n",
    ]


def build_sentinel_cell(target: str = "kaggle") -> dict[str, Any]:
    """Build the hardware sentinel code cell for Kaggle or Colab runtimes.

    torch is probed in a fresh interpreter, never imported into the notebook's own kernel
    process: the dependency cell installs the suite torch build afterwards, and a torch module
    already loaded in the kernel would then be stale (``operator torchvision::nms does not
    exist`` when suite code later imports torchvision there). No GPU stops the notebook.
    """
    text = _SENTINEL_TEXT["kaggle" if target == "kaggle" else "colab"]
    env_lines = _build_env_var_lines(_load_runtime_env())
    header = ["import os, subprocess, sys, time\n"]
    if target == "kaggle":
        header += [
            "# Session start for the training loop's deadline guard (first run only)\n",
            f"os.environ.setdefault({SESSION_START_ENV!r}, str(time.time()))\n",
        ]
    source = [
        *header,
        "# Runtime environment (LemGendary env-manager SSOT)\n",
        *env_lines,
        "print('[OK] [SENTINEL] Auditing Hardware Manifold...')\n",
        f"print({'[OK] [RECOMMENDED ACCELERATOR] ' + text.recommended!r})\n",
        "\n",
        "# torch is probed in a fresh interpreter, never imported into this kernel process.\n",
        "HW_PROBE = '''\n",
        *_hardware_probe_lines(text),
        "'''\n",
        "hw_probe = subprocess.run([sys.executable, '-c', HW_PROBE],\n",
        "                          capture_output=True, text=True)\n",
        "print(hw_probe.stdout, end='')\n",
        f"if hw_probe.returncode == {NO_GPU_EXIT_CODE}:\n",
        f"    raise RuntimeError({'[ABORT] No GPU accelerator detected. ' + text.abort!r})\n",
        "if hw_probe.returncode != 0:\n",
        "    print(hw_probe.stderr[-2000:])\n",
        (
            "    raise RuntimeError(f'[ABORT] Hardware probe failed with exit code "
            "{hw_probe.returncode}.')\n"
        ),
    ]
    return make_code_cell(source)
