"""Base definitions and protocols for LemGendary container reader plugins."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, NamedTuple, Protocol, runtime_checkable

# Shared read-only empty mapping used as the default ``Sample.metadata``.
_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})


class Sample(NamedTuple):
    """Immutable representation of an ingested dataset sample."""

    name: str
    image_bytes: bytes
    image_format: str
    target_bytes: bytes | None = None
    mask_bytes: bytes | None = None
    label: Any = None
    metadata: Mapping[str, Any] = _EMPTY_METADATA


@runtime_checkable
class ContainerReader(Protocol):
    """Protocol implemented by all LemGendary dataset container readers."""

    def __len__(self) -> int:
        """Total count of available samples in the active container split."""
        ...

    def __getitem__(self, index: int) -> Sample:
        """Fetch sample at the given index."""
        ...

    def close(self) -> None:
        """Release underlying memory maps, file descriptors, or cache handles."""
        ...
