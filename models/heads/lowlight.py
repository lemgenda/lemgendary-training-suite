"""Low-light enhancement output head."""

from models.heads.base_head import BaseRestorationHead


class LowLightHead(BaseRestorationHead):
    """Convolutional output head for low-light enhancement tasks."""
