"""Git clone and repository synchronization cell generator.

Clones the training suite and the environment manager on Kaggle and Colab without ever
writing the GitHub token to a file or a remote URL:

* the clone URLs are the plain ``<identity.github_url>/<folder>.git`` of the ecosystem file;
* the token reaches git only through ``GIT_ASKPASS`` (a helper script under the system temp
  folder that prints ``$LEMGENDARY_GIT_TOKEN``) and the environment of the git processes;
* the origin URL is reset to the plain URL on every run (older notebooks stored the token
  in it), and the helper folder is deleted afterwards;
* on Kaggle the checkouts live in ``/kaggle/temp``, outside ``/kaggle/working``, so no
  ``.git`` folder is ever saved as kernel output.
"""

from typing import Any

from training.ecosystem import ENV_MANAGER_FOLDER, github_repo_url

from ..platform_paths import SUITE_FOLDER, platform_paths
from .base import make_code_cell

_ASKPASS_SCRIPT = (
    '#!/bin/sh\\ncase "$1" in Username*) echo x-access-token ;; '
    '*) echo "$LEMGENDARY_GIT_TOKEN" ;; esac\\n'
)
"""``GIT_ASKPASS`` helper: user name for the user prompt, the token from the environment."""


def build_clone_cell(target: str = "kaggle") -> dict[str, Any]:
    """Build the git clone and environment synchronization code cell."""
    paths = platform_paths(target)
    secret_ui_action = "Kaggle Add-ons -> Secrets" if target == "kaggle" else "Colab Secrets"
    source = [
        "import os, shutil, stat, subprocess, sys, tempfile\n",
        f"suite_url = {github_repo_url(SUITE_FOLDER)!r}\n",
        f"env_mgr_url = {github_repo_url(ENV_MANAGER_FOLDER)!r}\n",
        f"checkout_dir = {paths.checkout_dir!r}\n",
        f"suite_path = {paths.suite_dir!r}\n",
        f"env_mgr_path = {paths.env_mgr_dir!r}\n",
        "os.makedirs(checkout_dir, exist_ok=True)\n",
        "# The suite runs from the checkout; outputs and the hub stay in the working dir.\n",
        f"os.environ['LEMGENDARY_WORKSPACE_ROOT'] = {paths.working_dir!r}\n",
        f"os.environ['LEMGENDARY_ECOSYSTEM_FILE'] = {paths.ecosystem_file!r}\n",
        *(
            [f"os.environ.setdefault('KAGGLE_WORKING_DIR', {paths.working_dir!r})\n"]
            if target == "kaggle"
            else []
        ),
        "\n",
        "token = os.environ.get('SUITE_PAT') or os.environ.get('GITHUB_PAT') or ''\n",
        "git_env = os.environ.copy()\n",
        "git_env['GIT_TERMINAL_PROMPT'] = '0'\n",
        "askpass_dir = tempfile.mkdtemp(prefix='lemgendary-askpass-')\n",
        "if token:\n",
        "    askpass = os.path.join(askpass_dir, 'askpass.sh')\n",
        "    with open(askpass, 'w', encoding='utf-8') as fh:\n",
        f"        fh.write('{_ASKPASS_SCRIPT}')\n",
        "    os.chmod(askpass, stat.S_IRWXU)\n",
        "    git_env['GIT_ASKPASS'] = askpass\n",
        "    git_env['LEMGENDARY_GIT_TOKEN'] = token\n",
        "    print('[AUTH] GitHub token supplied to git through GIT_ASKPASS (never stored).')\n",
        "else:\n",
        "    print('[WARNING] No SUITE_PAT or GITHUB_PAT secret; trying a public clone.')\n",
        "\n",
        "def _git(args, cwd=None):\n",
        (
            "    return subprocess.run(['git', *args], cwd=cwd, env=git_env, "
            "capture_output=True, text=True)\n"
        ),
        "\n",
        "def _sync_repo(url, path, marker):\n",
        "    valid = os.path.isdir(os.path.join(path, '.git')) and os.path.exists(\n",
        "        os.path.join(path, marker))\n",
        "    if not valid:\n",
        "        if os.path.exists(path):\n",
        "            shutil.rmtree(path)\n",
        "        res = _git(['clone', '--depth', '1', url, path])\n",
        "        if res.returncode != 0:\n",
        "            return f'clone failed: {res.stderr.strip()}'\n",
        "    # Reset the origin to the plain URL (older notebooks stored the token in it).\n",
        "    _git(['remote', 'set-url', 'origin', url], cwd=path)\n",
        "    if valid:\n",
        "        fetch = _git(['fetch', '--depth', '1', 'origin'], cwd=path)\n",
        "        if fetch.returncode != 0:\n",
        "            return f'fetch failed: {fetch.stderr.strip()}'\n",
        "        reset = _git(['reset', '--hard', 'origin/main'], cwd=path)\n",
        "        if reset.returncode != 0:\n",
        "            return f'reset failed: {reset.stderr.strip()}'\n",
        "    return None\n",
        "\n",
        "try:\n",
        (
            "    suite_error = _sync_repo(suite_url, suite_path, "
            "os.path.join('training', 'train.py'))\n"
        ),
        "    env_mgr_error = _sync_repo(env_mgr_url, env_mgr_path, 'ecosystem')\n",
        "finally:\n",
        "    try:\n",
        "        shutil.rmtree(askpass_dir)\n",
        "    except OSError as rm_err:\n",
        "        print(f'[WARNING] Could not remove {askpass_dir}: {rm_err}')\n",
        "\n",
        "if env_mgr_error:\n",
        "    print(f'[WARNING] env-manager {env_mgr_error}')\n",
        "if suite_error:\n",
        "    print(f'[ERROR] Suite {suite_error}')\n",
        (
            "    print('[ACTION REQUIRED] Add SUITE_PAT or GITHUB_PAT to "
            f"{secret_ui_action} with GitHub read permissions.')\n"
        ),
        "    sys.exit(1)\n",
        "print(f'[OK] Suite ready at {suite_path}')\n",
    ]
    return make_code_cell(source)
