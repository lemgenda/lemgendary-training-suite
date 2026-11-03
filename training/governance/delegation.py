"""Descriptor that forwards an attribute to a governor sub-state object.

``SmartTrainingGovernor`` exposes the fields of its ``CurriculumState``,
``SotaTracker`` and ``ThermalState`` members as flat attributes. Each one used
to be a hand-written property pair; this descriptor keeps that read/write
behaviour, including the type coercion applied on assignment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Delegated:
    """Read/write proxy for ``getattr(owner.<state>, <field>)``.

    ``cast`` (for example ``float`` or ``list``) is applied to assigned values.
    """

    def __init__(
        self,
        state: str,
        field: str | None = None,
        cast: Callable[[Any], Any] | None = None,
    ) -> None:
        self.state = state
        self.field = field or ""
        self.cast = cast
        self.__doc__: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        if not self.field:
            self.field = name
        self.__doc__ = f"Delegated ``{self.state}.{self.field}`` (read/write)."

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        return getattr(getattr(instance, self.state), self.field)

    def __set__(self, instance: Any, value: Any) -> None:
        if self.cast is not None:
            value = self.cast(value)
        setattr(getattr(instance, self.state), self.field, value)
