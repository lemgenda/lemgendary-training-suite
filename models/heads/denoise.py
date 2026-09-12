"""Denoise output head."""

from models.heads.base_head import BaseRestorationHead


class DenoiseHead(BaseRestorationHead):
    """Convolutional output head for image denoising tasks."""
