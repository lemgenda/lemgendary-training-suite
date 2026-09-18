"""Notebook builder implementations for target platforms.

Exports Kaggle and Google Colab automated training and usage notebook generators.
"""

from .base import (
    sync_manifold_notebooks,
    sync_workspace_training_notebook,
    write_notebook,
)
from .colab import (
    generate_colab_inference_notebook,
    generate_colab_training_notebook,
    generate_colab_usage_notebook,
)
from .kaggle import (
    generate_inference_notebook,
    generate_training_notebook,
    generate_usage_notebook,
)

__all__ = [
    "generate_inference_notebook",
    "generate_usage_notebook",
    "generate_training_notebook",
    "generate_colab_inference_notebook",
    "generate_colab_usage_notebook",
    "generate_colab_training_notebook",
    "write_notebook",
    "sync_manifold_notebooks",
    "sync_workspace_training_notebook",
]
