"""Notebook secrets cell: platform secrets into the process environment only (K-NB-6, K-AUTH-3).

The cell imports the platform's secrets client with a plain import, reads each secret through
one helper that catches only the client's "secret missing / not granted" errors, and exports:

* ``SUITE_PAT`` / ``GITHUB_PAT`` and ``GOOGLE_DRIVE`` as environment variables;
* ``KAGGLE_API_TOKEN`` only when it is a ``KGAT_`` API token (the Kaggle SDK and kagglehub read
  it from the environment); no ``kaggle.json`` or other Kaggle credential file is written and
  legacy keys are not used;
* ``GOOGLE_DRIVE_JSON`` as the service-account file ``~/.gdrive/service_account.json``
  (mode 0600, outside the kernel output) for the Google client libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training.cloud.credentials import KAGGLE_ACCESS_TOKEN_PREFIX

from .base import make_code_cell

KAGGLE_TOKEN_SECRET = "KAGGLE_API_TOKEN"
"""Secret holding the Kaggle API token (``KGAT_...``)."""
PAT_SECRETS: tuple[str, ...] = ("SUITE_PAT", "GITHUB_PAT")
"""GitHub token secrets; one of them is needed to clone the private repositories."""
OPTIONAL_SECRETS: tuple[str, ...] = ("GOOGLE_DRIVE", "GOOGLE_DRIVE_JSON")
"""Google Drive mirror secrets (optional)."""
SECRET_NAMES: tuple[str, ...] = (*PAT_SECRETS, KAGGLE_TOKEN_SECRET, *OPTIONAL_SECRETS)


@dataclass(frozen=True)
class SecretsClient:
    """How a platform's generated cell reads secrets."""

    imports: tuple[str, ...]
    getter: str
    missing_errors: str
    ui_hint: str


_CLIENTS: dict[str, SecretsClient] = {
    "kaggle": SecretsClient(
        imports=(
            "from kaggle_secrets import UserSecretsClient\n",
            "from kaggle_web_client import BackendError\n",
        ),
        getter="UserSecretsClient().get_secret",
        missing_errors="(BackendError, ConnectionError)",
        ui_hint="Add-ons -> Secrets",
    ),
    "colab": SecretsClient(
        imports=("from google.colab import userdata\n",),
        getter="userdata.get",
        missing_errors="(userdata.SecretNotFoundError, userdata.NotebookAccessError)",
        ui_hint="the Colab Secrets panel (key icon)",
    ),
}


def build_secrets_cell(target: str = "kaggle") -> dict[str, Any]:
    """Build the secrets cell for ``target`` (``kaggle``; any other target is Colab)."""
    client = _CLIENTS["kaggle" if target == "kaggle" else "colab"]
    source = [
        "import os\n",
        *client.imports,
        "\n",
        f"SECRET_NAMES = {SECRET_NAMES!r}\n",
        "\n",
        "def read_secret(name):\n",
        "    try:\n",
        f"        return {client.getter}(name) or None\n",
        f"    except {client.missing_errors} as exc:\n",
        "        print(f'[SECRETS] {name} not available ({type(exc).__name__}).')\n",
        "        return None\n",
        "\n",
        "secrets = {name: read_secret(name) for name in SECRET_NAMES}\n",
        f"for name in {(*PAT_SECRETS, 'GOOGLE_DRIVE')!r}:\n",
        "    if secrets[name]:\n",
        "        os.environ[name] = secrets[name]\n",
        "\n",
        f"kaggle_token = secrets[{KAGGLE_TOKEN_SECRET!r}]\n",
        f"if kaggle_token and kaggle_token.startswith({KAGGLE_ACCESS_TOKEN_PREFIX!r}):\n",
        f"    os.environ[{KAGGLE_TOKEN_SECRET!r}] = kaggle_token\n",
        "    os.environ.pop('KAGGLE_KEY', None)\n",
        "elif kaggle_token:\n",
        (
            f"    print('[REMEDY] {KAGGLE_TOKEN_SECRET} must be a Kaggle API token "
            f"({KAGGLE_ACCESS_TOKEN_PREFIX}...); legacy keys are not used.')\n"
        ),
        "else:\n",
        (
            f"    print('[REMEDY] Add the secret {KAGGLE_TOKEN_SECRET} (your Kaggle API "
            f"token, {KAGGLE_ACCESS_TOKEN_PREFIX}...) in {client.ui_hint}.')\n"
        ),
        "\n",
        "service_account = secrets['GOOGLE_DRIVE_JSON']\n",
        "if service_account:\n",
        "    sa_dir = os.path.expanduser('~/.gdrive')\n",
        "    os.makedirs(sa_dir, exist_ok=True)\n",
        "    sa_path = os.path.join(sa_dir, 'service_account.json')\n",
        "    with open(sa_path, 'w', encoding='utf-8') as sa_file:\n",
        "        sa_file.write(service_account)\n",
        "    os.chmod(sa_path, 0o600)\n",
        "    os.environ['GOOGLE_DRIVE_JSON'] = service_account\n",
        "    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sa_path\n",
        "\n",
        "active = [name for name, value in secrets.items() if value]\n",
        "print(f'[AUTH] Secrets available: {active}')\n",
        f"if not any(secrets[name] for name in {PAT_SECRETS!r}):\n",
        (
            f"    print('[REMEDY] Add SUITE_PAT or GITHUB_PAT in {client.ui_hint}; private "
            "repositories cannot be cloned without it.')\n"
        ),
    ]
    return make_code_cell(source)
