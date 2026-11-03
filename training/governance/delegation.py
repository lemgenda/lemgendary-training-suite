"""Descriptor that forwards an attribute to a governor sub-state object.

``SmartTrainingGovernor`` exposes the fields of its ``CurriculumState``,
``SotaTracker`` and ``ThermalState`` members as flat attributes. Each one used
to be a hand-written property pair; this descriptor keeps that read/write
behaviour, including the type coercion applied on assignment.

The descriptor is generic in the forwarded field's type, so ``Delegated[float]``
reads as ``float`` on an instance for the type checker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

if TYPE_CHECKING:
    from typing_extensions import Self

T = TypeVar("T")


class Delegated(Generic[T]):
    """Read/write proxy for ``getattr(owner.<state>, <field>)``.

    ``cast`` (for example ``float`` or ``list``) is applied to assigned values.
    """

    def __init__(
        self,
        state: str,
        field: str | None = None,
        cast: Callable[[Any], T] | None = None,
    ) -> None:
        self.state = state
        self.field = field or ""
        self.cast = cast
        self.__doc__: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        if not self.field:
            self.field = name
        self.__doc__ = f"Delegated ``{self.state}.{self.field}`` (read/write)."

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type | None = None) -> T: ...

    def __get__(self, instance: object | None, owner: type | None = None) -> Self | T:
        if instance is None:
            return self
        return getattr(getattr(instance, self.state), self.field)

    def __set__(self, instance: object, value: T) -> None:
        if self.cast is not None:
            value = self.cast(value)
        setattr(getattr(instance, self.state), self.field, value)
