"""Authoritative task type of a registry model entry (SSOT).

Every caller that needs a model's task type (the run spec, the governor and the
epoch quality scorer) resolves it here, so the three can never disagree.

Precedence: ``task_type``, then ``dataset_type``, then ``category``; the first
non-empty value wins. A list value (e.g. YOLO's ``[detection, classification,
pose]``) resolves to its first entry. A model entry declaring none of the three
fields resolves to :data:`DEFAULT_TASK_TYPE`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_TASK_TYPE = "restoration"

# Registry fields naming the task, highest precedence first.
TASK_TYPE_FIELDS = ("task_type", "dataset_type", "category")


def resolve_task_type(model_info: Mapping[str, Any]) -> str:
    """Return the model's task type (see the module docstring for the order)."""
    for field_name in TASK_TYPE_FIELDS:
        value = model_info.get(field_name)
        if isinstance(value, list):
            value = value[0] if value else None
        if value:
            return str(value)
    return DEFAULT_TASK_TYPE
