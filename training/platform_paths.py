"""Folders of the hosted notebook platforms (Kaggle, Colab): one place for K-SSOT-2.

Every module that needs a platform folder (``/kaggle/working``, ``/kaggle/input``, the clone
folder, ``/content``) reads it from here; the generated notebook cells render the same values.

* ``working_dir`` is the folder the platform keeps as output (Kaggle saves
  ``/kaggle/working`` as kernel output). ``LemGendaryModels`` lives there.
* ``checkout_dir`` is where the repositories are cloned. On Kaggle it is the session-only
  ``/kaggle/temp``, so no ``.git`` folder (config, hooks, packed refs) is ever published as
  kernel output.
* ``input_dir`` is where attached inputs are mounted (Kaggle datasets and models under
  ``/kaggle/input``; the mounted Google Drive on Colab).
* ``dataset_roots`` are the folders searched for compiled dataset manifolds.

The notebook cells export ``LEMGENDARY_WORKSPACE_ROOT`` (the working dir), so
:func:`training.utils.paths.get_models_hub_root` resolves the hub under the working dir although
the suite runs from the checkout dir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from training.ecosystem import ENV_MANAGER_FOLDER
from training.utils.paths import MODELS_FOLDER

__all__ = [
    "DATASETS_FOLDER",
    "KAGGLE_RUN_TYPE_ENV",
    "KAGGLE_WORKING_DIR_ENV",
    "MODELS_FOLDER",
    "SUITE_FOLDER",
    "PlatformPaths",
    "kaggle_paths",
    "on_kaggle",
    "platform_paths",
]

SUITE_FOLDER = "lemgendary-training-suite"
"""Folder (and repository) name of this suite."""
DATASETS_FOLDER = "LemGendaryDatasets"
"""Folder of the compiled dataset manifolds."""
KAGGLE_RUN_TYPE_ENV = "KAGGLE_KERNEL_RUN_TYPE"
"""Set by Kaggle in every kernel session (interactive and batch)."""
KAGGLE_WORKING_DIR_ENV = "KAGGLE_WORKING_DIR"
"""Exported by the generated notebook's clone cell (the Kaggle working dir)."""


@dataclass(frozen=True)
class PlatformPaths:
    """Folders of one notebook platform (POSIX paths rendered into cells)."""

    working_dir: str
    checkout_dir: str
    input_dir: str = ""
    dataset_roots: tuple[str, ...] = field(default_factory=tuple)

    @property
    def suite_dir(self) -> str:
        """Checkout of the training suite."""
        return f"{self.checkout_dir}/{SUITE_FOLDER}"

    @property
    def env_mgr_dir(self) -> str:
        """Checkout of the environment manager (owner of the ecosystem file)."""
        return f"{self.checkout_dir}/{ENV_MANAGER_FOLDER}"

    @property
    def ecosystem_file(self) -> str:
        """The ecosystem file inside the environment manager checkout."""
        return f"{self.env_mgr_dir}/ecosystem/ecosystem.json"

    @property
    def hub_root(self) -> str:
        """``LemGendaryModels`` under the working dir (kept as output)."""
        return f"{self.working_dir}/{MODELS_FOLDER}"

    @property
    def datasets_dir(self) -> str:
        """``LemGendaryDatasets`` under the working dir."""
        return f"{self.working_dir}/{DATASETS_FOLDER}"


_KAGGLE = PlatformPaths(
    working_dir="/kaggle/working",
    checkout_dir="/kaggle/temp",
    input_dir="/kaggle/input",
    dataset_roots=(f"/kaggle/working/{DATASETS_FOLDER}", "/kaggle/input"),
)
_COLAB = PlatformPaths(
    working_dir="/content",
    checkout_dir="/content",
    input_dir="/content/drive/MyDrive",
    dataset_roots=(f"/content/{DATASETS_FOLDER}", f"/content/drive/MyDrive/{DATASETS_FOLDER}"),
)
_PLATFORMS: dict[str, PlatformPaths] = {"kaggle": _KAGGLE, "colab": _COLAB}


def platform_paths(target: str) -> PlatformPaths:
    """Folders of ``target`` (``kaggle``; any other target is Colab, as in the cells)."""
    return _PLATFORMS["kaggle" if target == "kaggle" else "colab"]


def kaggle_paths() -> PlatformPaths:
    """Folders of a Kaggle kernel session."""
    return _KAGGLE


def on_kaggle() -> bool:
    """Whether this process runs inside a Kaggle kernel session.

    Kaggle sets ``KAGGLE_KERNEL_RUN_TYPE``; the generated notebook also exports
    ``KAGGLE_WORKING_DIR``. The working dir check covers kernels started outside the notebook.
    """
    if os.environ.get(KAGGLE_RUN_TYPE_ENV) or os.environ.get(KAGGLE_WORKING_DIR_ENV):
        return True
    return Path(_KAGGLE.working_dir).is_dir()
