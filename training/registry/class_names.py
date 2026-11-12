"""The ``class_names`` registry setting of classification models (single reader).

A classifier entry lists its output classes in id order::

    class_names: [drawings, hentai, neutral, porn, sexy]
    kwargs:
      num_classes: 5

Class ``i`` is the logit at index ``i``. ``kwargs.num_classes`` sizes the model head (the
factory passes ``kwargs`` to the constructor), so both must agree; a mismatch is rejected
here instead of surfacing as an out-of-range target in the loss. The label decoders of the
datasets, the model card and the export metadata read the names through
:func:`class_names`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CLASS_NAMES_KEY = "class_names"
NUM_CLASSES_KEY = "num_classes"
KWARGS_KEY = "kwargs"


class ClassNamesError(ValueError):
    """A registry ``class_names`` list that is malformed or disagrees with ``num_classes``."""


def _num_classes(model_info: Mapping[str, Any]) -> int | None:
    """``kwargs.num_classes`` of the entry, or ``None`` when it is not set."""
    kwargs = model_info.get(KWARGS_KEY) or {}
    if not isinstance(kwargs, Mapping):
        raise ClassNamesError(f"kwargs must be a mapping, got {type(kwargs).__name__}")
    value = kwargs.get(NUM_CLASSES_KEY)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ClassNamesError(f"kwargs.num_classes must be a positive integer, got {value!r}")
    return value


def class_names(model_info: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The entry's validated class names in id order; ``()`` when none are declared.

    ``ClassNamesError`` for a non-list, an empty, blank or duplicate name, or a length that
    differs from ``kwargs.num_classes``.
    """
    info = model_info or {}
    raw = info.get(CLASS_NAMES_KEY)
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ClassNamesError(f"class_names must be a non-empty list, got {raw!r}")
    names = tuple(str(name).strip() for name in raw)
    if any(not isinstance(name, str) for name in raw) or not all(names):
        raise ClassNamesError(f"class_names must be non-empty strings, got {raw!r}")
    if len(set(names)) != len(names):
        raise ClassNamesError(f"class_names must be unique, got {raw!r}")
    expected = _num_classes(info)
    if expected is not None and expected != len(names):
        raise ClassNamesError(
            f"class_names lists {len(names)} classes but kwargs.num_classes is {expected}"
        )
    return names
