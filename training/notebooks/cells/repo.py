"""Git clone and repository synchronization cell generator.

Handles authenticated repository cloning, shallow fetches, origin tracking,
and sibling environment manager synchronization on Kaggle and Colab.
"""

from typing import Any

from .base import make_code_cell


def build_clone_cell(target: str = "kaggle") -> dict[str, Any]:
    """Build the git clone and environment synchronization code cell."""
    if target == "kaggle":
        suite_path = "/kaggle/working/lemgendary-training-suite"
        env_mgr_path = "/kaggle/working/lemgendary-env-manager"
        secret_ui_action = "Kaggle Add-ons -> Secrets"
    else:
        suite_path = "/content/lemgendary-training-suite"
        env_mgr_path = "/content/lemgendary-env-manager"
        secret_ui_action = "Colab Secrets"

    source = [
        "import os, subprocess, shutil, sys\n",
        "from urllib.parse import quote as _url_quote\n",
        "repo_url = 'https://github.com/lemgenda/lemgendary-training-suite.git'\n",
        f"suite_path = '{suite_path}'\n",
        "pat = os.environ.get('SUITE_PAT', os.environ.get('GITHUB_PAT', ''))\n",
        "if pat:\n",
        "    _safe_pat = _url_quote(pat, safe='')\n",
        "    auth_url = repo_url.replace('https://', f'https://x-access-token:{_safe_pat}@')\n",
        "    print(f'[AUTH] Using {\"SUITE_PAT\" if os.environ.get(\"SUITE_PAT\") else \"GITHUB_PAT\"} for cloning...')\n",
        "else:\n",
        "    print('[WARNING] No PAT found in environment. Attempting public clone (will fail for private repos)...')\n",
        f"    print('[ACTION REQUIRED] If clone fails, add SUITE_PAT or GITHUB_PAT to {secret_ui_action}.')\n",
        "    auth_url = repo_url\n",
        "\n",
        "env = os.environ.copy()\n",
        "env['GIT_TERMINAL_PROMPT'] = '0'\n",
        "\n",
        "def _is_valid_repo(path, marker=None):\n",
        "    if not os.path.isdir(path):\n",
        "        return False\n",
        "    if not os.path.isdir(os.path.join(path, '.git')):\n",
        "        return False\n",
        "    if marker and not os.path.exists(os.path.join(path, marker)):\n",
        "        return False\n",
        "    return True\n",
        "\n",
        "if not _is_valid_repo(suite_path, marker=os.path.join('training', 'train.py')):\n",
        "    if os.path.exists(suite_path):\n",
        "        print('[SUITE] Removing incomplete previous clone...')\n",
        "        shutil.rmtree(suite_path, ignore_errors=True)\n",
        "    print('[SUITE] Initializing LemGendary Training Suite...')\n",
        "    res = subprocess.run(['git', 'clone', '--depth', '1', auth_url, suite_path], capture_output=True, text=True, env=env)\n",
        "    if res.returncode == 0:\n",
        "        print('[OK] Suite cloned.')\n",
        "    else:\n",
        "        print(f'[ERROR] Clone failed: {res.stderr.strip()}')\n",
        "        if '403' in res.stderr or '401' in res.stderr or 'terminal prompts disabled' in res.stderr:\n",
        f"            print('[ACTION REQUIRED] Add SUITE_PAT or GITHUB_PAT to {secret_ui_action} with GitHub read permissions.')\n",
        "        sys.exit(1)\n",
        "else:\n",
        "    print('[OK] Suite resident. Syncing origin and pulling latest...')\n",
        "    subprocess.run(['git', 'remote', 'set-url', 'origin', auth_url], cwd=suite_path, env=env)\n",
        "    fetch = subprocess.run(['git', 'fetch', '--depth', '1', 'origin'], cwd=suite_path, env=env, capture_output=True, text=True)\n",
        "    if fetch.returncode != 0:\n",
        "        print(f'[WARNING] git fetch failed: {fetch.stderr.strip()}')\n",
        "    reset = subprocess.run(['git', 'reset', '--hard', 'origin/main'], cwd=suite_path, env=env, capture_output=True, text=True)\n",
        "    if reset.returncode != 0:\n",
        "        print(f'[WARNING] git reset failed: {reset.stderr.strip()}')\n",
        "\n",
        "env_mgr_url = 'https://github.com/lemgenda/lemgendary-env-manager.git'\n",
        f"env_mgr_path = '{env_mgr_path}'\n",
        "if pat:\n",
        "    env_mgr_auth = env_mgr_url.replace('https://', f'https://x-access-token:{_url_quote(pat, safe=\"\")}@')\n",
        "else:\n",
        "    env_mgr_auth = env_mgr_url\n",
        "\n",
        "if not _is_valid_repo(env_mgr_path):\n",
        "    if os.path.exists(env_mgr_path):\n",
        "        shutil.rmtree(env_mgr_path, ignore_errors=True)\n",
        "    res_mgr = subprocess.run(['git', 'clone', '--depth', '1', env_mgr_auth, env_mgr_path], capture_output=True, text=True, env=env)\n",
        "    if res_mgr.returncode != 0:\n",
        "        print(f'[WARNING] env-manager clone failed: {res_mgr.stderr.strip()}')\n",
        "else:\n",
        "    subprocess.run(['git', 'pull'], cwd=env_mgr_path, env=env, capture_output=True)\n",
    ]

    return make_code_cell(source)
