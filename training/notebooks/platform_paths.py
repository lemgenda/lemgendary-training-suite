"""Where generated cloud notebooks keep repositories, outputs and the model hub.

One place for the platform folders the notebook cells render (K-SSOT-2 for these cells):

* ``working_dir`` is the folder the platform keeps as output (Kaggle saves
  ``/kaggle/working`` as kernel output). ``LemGendaryModels`` lives there.
* ``checkout_dir`` is where the repositories are cloned. On Kaggle it is the session-only
  ``/kaggle/temp``, so no ``.git`` folder (config, hooks, packed refs) is ever published as
  kernel output.

The cells export ``LEMGENDARY_WORKSPACE_ROOT`` (the working dir), ``LEMGENDARY_ECOSYSTEM_FILE``
and ``KAGGLE_WORKING_DIR`` so the suite resolves the hub under the working dir although it
runs from the checkout dir.
"""

from __future__ import annotations

from dataclasses import dataclass

from training.ecosystem import ENV_MANAGER_FOLDER

SUITE_FOLDER = "lemgendary-training-suite"
"""Folder (and repository) name of this suite."""
MODELS_FOLDER = "LemGendaryModels"
"""Folder of the trained models hub."""


@dataclass(frozen=True)
class PlatformPaths:
    """Folders of one notebook platform (POSIX paths rendered into cells)."""

    working_dir: str
    checkout_dir: str

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


_PLATFORMS: dict[str, PlatformPaths] = {
    "kaggle": PlatformPaths(working_dir="/kaggle/working", checkout_dir="/kaggle/temp"),
    "colab": PlatformPaths(working_dir="/content", checkout_dir="/content"),
}


def platform_paths(target: str) -> PlatformPaths:
    """Folders of ``target`` (``kaggle``; any other target is Colab, as in the cells)."""
    return _PLATFORMS["kaggle" if target == "kaggle" else "colab"]
