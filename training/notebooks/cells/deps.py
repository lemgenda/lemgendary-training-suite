"""Dependency installation and hardware acceleration probing cell generator.

Probes CUDA / ROCm / CPU hardware accelerators and installs package manifests
from lemgendary-env-manager or lemgendary-training-suite. torch is never imported into the
notebook's own kernel process: the accelerator fallback probe and the post-install version
probe run in a fresh interpreter, so they see the build this cell installed.
"""

from typing import Any

from training.platform_paths import SUITE_FOLDER, platform_paths

from ..torch_build import CLOUD_CUDA_BUILD, TORCH_FAMILY
from .base import make_code_cell

_SECRETS_PLACE = {"kaggle": "Kaggle Secrets", "colab": "Colab Secrets"}
_SECRETS_FIX = {
    "kaggle": "Go to Kaggle Notebook top bar -> Add-ons -> Secrets -> Add SUITE_PAT or "
    "GITHUB_PAT with your GitHub token.",
    "colab": "Go to Colab Secrets and add SUITE_PAT or GITHUB_PAT with your GitHub token.",
}


def _req_candidates(target: str) -> list[str]:
    """Source lines listing the requirement manifests, most specific first."""
    paths = platform_paths(target)
    candidates = [
        f"{paths.env_mgr_dir}/requirements/requirements-training.txt",
        f"{paths.env_mgr_dir}/requirements/{SUITE_FOLDER}.requirements.txt",
        f"{paths.suite_dir}/requirements.txt",
    ]
    return ["req_candidates = [\n", *(f"    {c!r},\n" for c in candidates), "]\n"]


_ACCELERATOR_LINES = [
    "import os, sys, subprocess, platform, shutil, tempfile\n",
    "print('[ENV] Probing hardware accelerator...')\n",
    "\n",
    "torch_index = 'https://download.pytorch.org/whl/cpu'\n",
    "accel_type = 'cpu'\n",
    "if (os.environ.get('TPU_NAME') or os.environ.get('KAGGLE_TPU_NAME')\n"
    "        or os.path.exists('/dev/vfio/0')):\n",
    "    accel_type = 'tpu'\n",
    "    os.environ.setdefault('PJRT_DEVICE', 'TPU')\n",
    "    os.environ.setdefault('XLA_USE_BF16', '1')\n",
    "    print('[ENV] TPU VM detected; preserving preinstalled torch/torch_xla stack.')\n",
    "elif shutil.which('nvidia-smi'):\n",
    f"    torch_index = {CLOUD_CUDA_BUILD.index_url!r}\n",
    f"    accel_type = {CLOUD_CUDA_BUILD.accelerator!r}\n",
    "elif shutil.which('rocm-smi'):\n",
    "    torch_index = 'https://download.pytorch.org/whl/rocm6.1'\n",
    "    accel_type = 'rocm6.1'\n",
    "else:\n",
    "    # A preinstalled torch is asked in a fresh interpreter, never imported here.\n",
    "    torch_probe = subprocess.run(\n",
    "        [sys.executable, '-c',\n",
    "         'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"\"); '\n",
    '         \'print(getattr(torch.version, "hip", None) or "")\'],\n',
    "        capture_output=True, text=True)\n",
    "    if torch_probe.returncode != 0:\n",
    "        print('[ENV] torch is not installed yet; using the CPU index.')\n",
    "    else:\n",
    "        probed = [v.strip() for v in torch_probe.stdout.splitlines()] + ['', '']\n",
    "        cuda_v, hip_v = probed[0], probed[1]\n",
    "        if cuda_v:\n",
    "            cuda_tag = 'cu' + ''.join(cuda_v.split('.')[:2])\n",
    "            torch_index = f'https://download.pytorch.org/whl/{cuda_tag}'\n",
    "            accel_type = f'cuda_{cuda_tag}'\n",
    "        elif hip_v:\n",
    "            rocm_tag = 'rocm' + '.'.join(hip_v.split('.')[:2])\n",
    "            torch_index = f'https://download.pytorch.org/whl/{rocm_tag}'\n",
    "            accel_type = f'rocm_{rocm_tag}'\n",
    "\n",
]

_DIRECTML_LINES = [
    "if accel_type == 'cpu' and platform.system() == 'Windows':\n",
    "    torch_index = 'https://download.pytorch.org/whl/cpu'\n",
    "    accel_type = 'directml'\n",
    "\n",
]

_TORCH_VERSION_PROBE_LINES = [
    "        # The installed torch is asked in a fresh interpreter: a torch module imported\n",
    "        # in this kernel process before the install would report the old build.\n",
    "        version_probe = subprocess.run(\n",
    "            [sys.executable, '-c',\n",
    "             'import torch; print(torch.__version__, \"on\", '\n",
    '             \'"CUDA " + torch.version.cuda if torch.cuda.is_available() else "CPU")\'],\n',
    "            capture_output=True, text=True)\n",
    "        if version_probe.returncode == 0:\n",
    "            print(f'[OK] torch {version_probe.stdout.strip()}')\n",
    "        else:\n",
    "            print(f'[WARNING] Installed torch could not be probed (exit code '\n",
    "                  f'{version_probe.returncode}): {version_probe.stderr[-2000:]}')\n",
]


_TORCH_BUILD_LINES = [
    "    install_req, constraint_args = req_path, []\n",
    "    if torch_pins:\n",
    "        # The torch family comes from the build's own index only; the manifest's torch\n",
    "        # pins name the host build, so they are dropped and the rest is constrained.\n",
    "        print(f'[ENV] Torch build: {torch_pins}')\n",
    "        torch_res = subprocess.run(\n",
    "            [sys.executable, '-m', 'pip', 'install', '-q', '--index-url', torch_index,\n",
    "             *torch_pins], capture_output=True, text=True)\n",
    "        if torch_res.returncode != 0:\n",
    "            print('[WARNING] Torch build installation finished with non-zero exit code.')\n",
    "            print(torch_res.stderr[-2000:])\n",
    "        pin_dir = tempfile.mkdtemp(prefix='lemgendary-pins-')\n",
    "        install_req = os.path.join(pin_dir, 'requirements-without-torch.txt')\n",
    "        constraints = os.path.join(pin_dir, 'torch-constraints.txt')\n",
    "        with open(req_path, encoding='utf-8') as src:\n",
    "            req_lines = src.readlines()\n",
    "        with open(install_req, 'w', encoding='utf-8') as dst:\n",
    "            for line in req_lines:\n",
    "                name = line.split('#')[0].strip()\n",
    "                for sep in '=<>!~;[ ':\n",
    "                    name = name.split(sep)[0]\n",
    "                if name.lower() not in torch_family:\n",
    "                    dst.write(line)\n",
    "        with open(constraints, 'w', encoding='utf-8') as fh:\n",
    "            fh.write(''.join(pin + chr(10) for pin in torch_pins))\n",
    "        constraint_args = ['-c', constraints]\n",
]


def _install_lines(target: str) -> list[str]:
    """Install the requirement manifest against the accelerator's torch index.

    On the CUDA index of :data:`CLOUD_CUDA_BUILD` the torch family is installed first from
    that index alone (:data:`_TORCH_BUILD_LINES`), then the manifest without it, constrained
    to those builds so no PyPI wheel replaces them.
    """
    upgrade = "'--upgrade', 'kaggle>=2.2.5', " if target == "kaggle" else ""
    return [
        f"torch_family = {TORCH_FAMILY!r}\n",
        (
            f"torch_pins = {list(CLOUD_CUDA_BUILD.pins)!r} "
            f"if torch_index == {CLOUD_CUDA_BUILD.index_url!r} else []\n"
        ),
        "if req_path:\n",
        "    print(f'[ENV] Manifest: {req_path}')\n",
        *_TORCH_BUILD_LINES,
        "    res = subprocess.run(\n",
        f"        [sys.executable, '-m', 'pip', 'install', '-q', {upgrade}\n",
        "         '--extra-index-url', torch_index,\n",
        "         '--upgrade-strategy', 'only-if-needed',\n",
        "         *constraint_args, '-r', install_req],\n",
        "        capture_output=True, text=True)\n",
        "    if res.returncode == 0:\n",
        "        print('[OK] Environment Ready.')\n",
        *_TORCH_VERSION_PROBE_LINES,
        "    else:\n",
        "        print('[WARNING] Dependency installation finished with non-zero exit code.')\n",
        "        if res.stderr:\n",
        "            print(res.stderr[-2000:])\n",
    ]


def _missing_manifest_lines(target: str) -> list[str]:
    """Explain a missing manifest and install the container streaming fallback."""
    place = _SECRETS_PLACE.get(target, _SECRETS_PLACE["colab"])
    fix = _SECRETS_FIX.get(target, _SECRETS_FIX["colab"])
    return [
        "else:\n",
        "    print('[ERROR] Could not open requirements file: No such file or directory')\n",
        "    print(\"[REMEDY] Ensure 'requirements.txt' exists in the root of the repository.\")\n",
        (
            "    print('[ACTION REQUIRED] Suite clone failed in Step 3 because "
            f"SUITE_PAT/GITHUB_PAT is missing from {place}.')\n"
        ),
        f"    print({'[ACTION REQUIRED] Fix: ' + fix!r})\n",
        "    print('[INFO] Installing core container streaming dependencies as fallback...')\n",
        (
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', "
            "'webdataset==1.0.2', 'litdata>=0.2.0,<1.0.0', 'pyarrow>=25.0.1', "
            "'zstandard>=0.23.0'])\n"
        ),
    ]


def build_install_cell(target: str = "kaggle") -> dict[str, Any]:
    """Build the dependency installation and hardware probe code cell."""
    source = [
        *_ACCELERATOR_LINES,
        *(_DIRECTML_LINES if target == "kaggle" else []),
        "print(f'[ENV] Accelerator: {accel_type}')\n",
        "print(f'[ENV] PyTorch index: {torch_index}')\n",
        "\n",
        *_req_candidates(target),
        "req_path = next((p for p in req_candidates if os.path.exists(p)), None)\n",
        "\n",
        *_install_lines(target),
        *_missing_manifest_lines(target),
    ]
    return make_code_cell(source)
