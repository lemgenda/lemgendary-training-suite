"""Notebook Service for LemGendary Model Training Suite.

Coordinates automated Jupyter notebook generation for Kaggle and Colab cloud environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.notebooks import (
    generate_colab_inference_notebook,
    generate_colab_training_notebook,
    generate_colab_usage_notebook,
    generate_inference_notebook,
    generate_training_notebook,
    generate_usage_notebook,
    load_registry,
)
from training.utils.paths import get_project_root


class NotebookService:
    """Generates standalone execution notebooks for training, inference, and usage."""

    SUPPORTED_PLATFORMS: list[str] = ["kaggle", "colab"]
    SUPPORTED_KINDS: list[str] = ["training", "inference", "usage"]

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.default_output_dir = self.project_root / "notebooks"

    def list_supported_models(self) -> list[str]:
        """Return list of model keys available for notebook synthesis."""
        registry = load_registry()
        return sorted([k for k in registry.keys() if not k.startswith("_")])

    def generate_notebooks(
        self,
        model_key: str,
        platform: str = "kaggle",
        kinds: list[str] | None = None,
        output_dir: Path | str | None = None,
    ) -> dict[str, str]:
        """Generate notebooks for the given model key and platform.

        Args:
            model_key: Target model key registered in unified models.
            platform: Cloud platform ('kaggle' or 'colab').
            kinds: Subset of ['training', 'inference', 'usage']. Defaults to all.
            output_dir: Destination directory for generated .ipynb files.

        Returns:
            dict[str, str]: Map of notebook kind to output file path.
        """
        if platform not in self.SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Unsupported platform '{platform}'. Supported: {self.SUPPORTED_PLATFORMS}"
            )

        target_kinds = kinds or list(self.SUPPORTED_KINDS)
        for k in target_kinds:
            if k not in self.SUPPORTED_KINDS:
                raise ValueError(
                    f"Unsupported notebook kind '{k}'. Supported: {self.SUPPORTED_KINDS}"
                )

        dest_dir = Path(output_dir).resolve() if output_dir else self.default_output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, str] = {}

        if platform == "kaggle":
            if "training" in target_kinds:
                out = generate_training_notebook(model_key, output_dir=dest_dir)
                results["kaggle_training"] = str(out)
            if "inference" in target_kinds:
                out = generate_inference_notebook(model_key, output_dir=dest_dir)
                results["kaggle_inference"] = str(out)
            if "usage" in target_kinds:
                out = generate_usage_notebook(model_key, output_dir=dest_dir)
                results["kaggle_usage"] = str(out)
        elif platform == "colab":
            if "training" in target_kinds:
                out = generate_colab_training_notebook(model_key, output_dir=dest_dir)
                results["colab_training"] = str(out)
            if "inference" in target_kinds:
                out = generate_colab_inference_notebook(model_key, output_dir=dest_dir)
                results["colab_inference"] = str(out)
            if "usage" in target_kinds:
                out = generate_colab_usage_notebook(model_key, output_dir=dest_dir)
                results["colab_usage"] = str(out)

        return results
