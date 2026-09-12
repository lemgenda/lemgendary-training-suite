"""Dehaze output head."""

from models.heads.base_head import BaseRestorationHead


class DehazeHead(BaseRestorationHead):
    """Convolutional output head for dehazing tasks."""
