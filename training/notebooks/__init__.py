"""LemGendary Model Training Suite — Notebook Generation Subsystem.

Provides modular cell generators, platform builders, and registry resolution
for automated cloud training and model inference notebooks.
"""

from .builders.colab import (
    generate_colab_inference_notebook,
    generate_colab_training_notebook,
    generate_colab_usage_notebook,
)
from .builders.kaggle import (
    generate_inference_notebook,
    generate_training_notebook,
    generate_usage_notebook,
)
from .registry import (
    ModelNotebookMeta,
    load_registry,
    resolve_model_metadata,
)

__all__ = [
    "generate_inference_notebook",
    "generate_usage_notebook",
    "generate_training_notebook",
    "generate_colab_inference_notebook",
    "generate_colab_usage_notebook",
    "generate_colab_training_notebook",
    "resolve_model_metadata",
    "load_registry",
    "ModelNotebookMeta",
]
