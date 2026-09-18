"""Runtime environment and hardware sentinel cell generator.

Embeds runtime environment configuration from LemGendary Environment Manager
and generates hardware probe sentinel cells.
"""

from pathlib import Path
from typing import Any

from .base import make_code_cell

_RUNTIME_ENV_FALLBACK: dict[str, str] = {
    "PYTHONUTF8": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

_BLOCKED_RUNTIME_ENV: set[str] = {
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
        here.parent.parent.parent.parent.parent / "lemgendary-env-manager" / "requirements" / "runtime_env.yaml",
        here.parent.parent.parent.parent / "lemgendary-env-manager" / "requirements" / "runtime_env.yaml",
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
        import yaml
        from yaml import YAMLError
    except ImportError:
        return dict(_RUNTIME_ENV_FALLBACK)

    try:
        data = yaml.safe_load(raw) or {}
    except YAMLError:
        return dict(_RUNTIME_ENV_FALLBACK)

    result: dict[str, str] = {}
    for name, spec in (data.get("variables") or {}).items():
        if isinstance(spec, dict) and "value" in spec:
            result[str(name)] = str(spec["value"])

    result = {k: v for k, v in result.items() if k not in _BLOCKED_RUNTIME_ENV}
    result.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    return result if result else dict(_RUNTIME_ENV_FALLBACK)


def _build_env_var_lines(env_vars: dict[str, str]) -> list[str]:
    """Return Python source lines that assign each env var, sorted by name."""
    lines: list[str] = []
    for name in sorted(env_vars.keys()):
        value = env_vars[name].replace("\\", "\\\\").replace("'", "\\'")
        lines.append(f"os.environ['{name}'] = '{value}'\n")
    return lines


def build_sentinel_cell(target: str = "kaggle") -> dict[str, Any]:
    """Build the hardware sentinel code cell for Kaggle or Colab runtimes."""
    env_vars = _load_runtime_env()
    env_lines = _build_env_var_lines(env_vars)

    if target == "kaggle":
        accel_str = "GPU T4 x2 (30GB total VRAM) [Recommended]"
        source = [
            "import os, sys, subprocess\n",
            "# Runtime environment (LemGendary env-manager SSOT)\n",
        ] + env_lines + [
            "print('[OK] [SENTINEL] Auditing Hardware Manifold...')\n",
            f"print('[OK] [RECOMMENDED ACCELERATOR] Kaggle: {accel_str}')\n",
            "\n",
            "import torch\n",
            "_compat = True\n",
            "if torch.cuda.is_available():\n",
            "    _gpu_name = torch.cuda.get_device_name(0)\n",
            "    _cap = torch.cuda.get_device_capability(0)\n",
            "    _archs = getattr(torch.cuda, 'get_arch_list', lambda: [])()\n",
            "    _compat = any(f'{_cap[0]}.{_cap[1]}' in a or f'sm_{_cap[0]}{_cap[1]}' in a for a in _archs)\n",
            "    try:\n",
            "        _t = torch.ones(1, device='cuda') + 1.0\n",
            "        torch.cuda.synchronize()\n",
            "        del _t\n",
            "    except Exception as _k_err:\n",
            "        if any(_e in str(_k_err) for _e in ['no kernel image', 'cudaErrorNoKernelImageForDevice', 'capability']):\n",
            "            _compat = False\n",
            "    if not _compat:\n",
            "        print(f'[CRITICAL ERROR] [HARDWARE] NVIDIA {_gpu_name} (sm_{_cap[0]}{_cap[1]}) has no kernel images in current PyTorch build!')\n",
            "        print('[ACTION REQUIRED] Switch Kaggle Accelerator to GPU T4 x2 in Notebook Settings (right panel).')\n",
            "        print('[AUTO-FIX] Alternatively run: !pip install --force-reinstall torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121')\n",
            "    else:\n",
            "        print(f'[OK] [HARDWARE] NVIDIA {_gpu_name} (sm_{_cap[0]}{_cap[1]}) validated & ready.')\n",
            "\n",
            "if not torch.cuda.is_available():\n",
            "    print('[CRITICAL ERROR] [HARDWARE] NO GPU DETECTED! Training cannot proceed on CPU.')\n",
            "    print('[ACTION REQUIRED] Enable GPU Accelerator before running this notebook:')\n",
            f"    print('   -> Kaggle: Right Panel -> Session Options -> Accelerator -> {accel_str}')\n",
            "    raise RuntimeError('[ABORT] No GPU accelerator detected. Enable GPU in Session Options and re-run from the top.')\n",
            "else:\n",
            "    props = torch.cuda.get_device_properties(0)\n",
            "    cap = torch.cuda.get_device_capability(0)\n",
            "    _status_tag = '[OK] [ACTIVE]' if _compat else '[WARNING] [INCOMPATIBLE ACCELERATOR]'\n",
            "    print(f'{_status_tag} {props.name} (Compute Capability sm_{cap[0]}{cap[1]})')\n",
            "    print(f'[OK] [VRAM] {props.total_memory / 1024**3:.1f} GB')\n",
            "    if torch.cuda.device_count() > 1:\n",
            "        print(f'[OK] [MULTI-GPU] Detected {torch.cuda.device_count()} active GPUs.')\n",
            "    if props.total_memory / 1024**3 < 10.0:\n",
            "        print('[WARNING] Low VRAM detected. Suite will enable Survival Profiles automatically.')\n",
        ]
    else:
        source = [
            "import os, sys, subprocess\n",
            "# Runtime environment (LemGendary env-manager SSOT)\n",
        ] + env_lines + [
            "print('[OK] [SENTINEL] Auditing Hardware Manifold...')\n",
            "print('[OK] [RECOMMENDED ACCELERATOR] Google Colab: T4 GPU (or A100/L4 with Pro)')\n",
            "\n",
            "import torch\n",
            "_compat = True\n",
            "if torch.cuda.is_available():\n",
            "    _gpu_name = torch.cuda.get_device_name(0)\n",
            "    _cap = torch.cuda.get_device_capability(0)\n",
            "    _archs = getattr(torch.cuda, 'get_arch_list', lambda: [])()\n",
            "    _compat = any(f'{_cap[0]}.{_cap[1]}' in a or f'sm_{_cap[0]}{_cap[1]}' in a for a in _archs)\n",
            "    try:\n",
            "        _t = torch.ones(1, device='cuda') + 1.0\n",
            "        torch.cuda.synchronize()\n",
            "        del _t\n",
            "    except Exception as _k_err:\n",
            "        if any(_e in str(_k_err) for _e in ['no kernel image', 'cudaErrorNoKernelImageForDevice', 'capability']):\n",
            "            _compat = False\n",
            "    if not _compat:\n",
            "        print(f'[CRITICAL ERROR] [HARDWARE] NVIDIA {_gpu_name} (sm_{_cap[0]}{_cap[1]}) has no kernel images in current PyTorch build!')\n",
            "        print('[ACTION REQUIRED] Switch Colab Runtime to T4 GPU (Runtime -> Change runtime type -> T4 GPU).')\n",
            "        print('[AUTO-FIX] Alternatively run: !pip install --force-reinstall torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121')\n",
            "    else:\n",
            "        print(f'[OK] [HARDWARE] NVIDIA {_gpu_name} (sm_{_cap[0]}{_cap[1]}) validated & ready.')\n",
            "\n",
            "if not torch.cuda.is_available():\n",
            "    print('[CRITICAL ERROR] [HARDWARE] NO GPU DETECTED! Training cannot proceed on CPU.')\n",
            "    print('[ACTION REQUIRED] Enable GPU Accelerator before running this notebook:')\n",
            "    print('   -> Colab:  Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU')\n",
            "    raise RuntimeError('[ABORT] No GPU accelerator detected. Enable GPU in Colab Runtime settings and re-run from the top.')\n",
            "else:\n",
            "    props = torch.cuda.get_device_properties(0)\n",
            "    cap = torch.cuda.get_device_capability(0)\n",
            "    _status_tag = '[OK] [ACTIVE]' if _compat else '[WARNING] [INCOMPATIBLE ACCELERATOR]'\n",
            "    print(f'{_status_tag} {props.name} (Compute Capability sm_{cap[0]}{cap[1]})')\n",
            "    print(f'[OK] [VRAM] {props.total_memory / 1024**3:.1f} GB')\n",
            "    if torch.cuda.device_count() > 1:\n",
            "        print(f'[OK] [MULTI-GPU] Detected {torch.cuda.device_count()} active GPUs.')\n",
            "    if props.total_memory / 1024**3 < 10.0:\n",
            "        print('[WARNING] Low VRAM detected. Suite will enable Survival Profiles automatically.')\n",
        ]

    return make_code_cell(source)
