"""Model constructor configs: frozen dataclasses built from registry ``model_kwargs``.

A checkpoint stores its model's constructor keywords as ``model_kwargs`` and
``models.factory.get_model`` passes the registry ``kwargs`` the same way, so a
model whose hyper-parameters live in a config dataclass still accepts them as
flat keywords. ``resolve_config`` turns those keywords into the dataclass.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

ConfigT = TypeVar("ConfigT", bound="DataclassInstance")


def resolve_config(
    config_cls: type[ConfigT], config: ConfigT | None, kwargs: dict[str, Any]
) -> ConfigT:
    """Return ``config``, or a ``config_cls`` built from the matching ``kwargs``.

    Keywords that name a field of ``config_cls`` fill that field; a ``None``
    value keeps the field default. Other keywords are left for the caller.
    Passing ``config`` together with field keywords is ambiguous and raises
    ``TypeError``.
    """
    names = {field.name for field in dataclasses.fields(config_cls)}
    given = {key: value for key, value in kwargs.items() if key in names}
    if config is not None:
        if given:
            raise TypeError(
                f"{config_cls.__name__} given both as config and as keywords: {sorted(given)}"
            )
        return config
    return config_cls(**{key: value for key, value in given.items() if value is not None})
