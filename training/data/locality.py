"""Shard-locality orderings: windowed epoch shuffles and block-wise curriculum subsets.

A uniform shuffle over a sharded WebDataset seeks into a different tar shard on almost
every sample, so the OS page cache and readahead are wasted and every worker keeps many
shards open. Two orderings keep reads local while staying deterministic:

* :func:`shard_window_order` (epoch order): shuffle the shards for the epoch, cut the
  shard order into windows of ``window_shards`` shards and shuffle the samples inside
  each window. Batches still mix ``window_shards`` shards, but a window's shards are
  read together and then left.
* :func:`block_fraction_subset` (curriculum fraction): split the sample list into
  contiguous blocks inside each shard, shuffle the blocks with a fixed seed and take
  whole blocks from the front. A fraction then touches a share of the shards' blocks
  instead of a few samples from every shard, and the subsets stay nested: a larger
  fraction is a prefix extension of a smaller one (Monotonic Ratchet).

Both are off by default (``0``), which keeps the uniform per-sample orderings. The
settings live in the optional ``data.locality`` config section:

* ``window_shards`` (int >= 0, default 0): shards per shuffle window of the train
  sampler; ``0`` keeps the uniform epoch permutation.
* ``fraction_block_size`` (int >= 0, default 0): samples per block of the curriculum
  fraction subset; ``0`` keeps the per-sample subset.

This module is torch-free so the settings can be validated without the training stack.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

CONFIG_SECTION = "data"
LOCALITY_KEY = "locality"
FRACTION_SEED = 42
"""Seed of the curriculum subset order (the per-sample subset uses the same)."""

T = TypeVar("T")


@dataclass(frozen=True)
class LocalityConfig:
    """Validated ``data.locality`` settings (see the module docstring)."""

    window_shards: int = 0
    fraction_block_size: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> LocalityConfig:
        """Build the settings from the parsed ``config.yaml``.

        Raises:
            TypeError: a section is not a mapping or a value is not an integer.
            ValueError: a value is negative.
        """
        data = (config or {}).get(CONFIG_SECTION) or {}
        if not isinstance(data, Mapping):
            raise TypeError(f"'{CONFIG_SECTION}' must be a mapping, got {type(data).__name__}")
        section = data.get(LOCALITY_KEY) or {}
        if not isinstance(section, Mapping):
            raise TypeError(
                f"'{CONFIG_SECTION}.{LOCALITY_KEY}' must be a mapping, got {type(section).__name__}"
            )
        return cls(
            window_shards=_non_negative_int(section, "window_shards"),
            fraction_block_size=_non_negative_int(section, "fraction_block_size"),
        )


def _non_negative_int(section: Mapping[str, Any], key: str) -> int:
    raw = section.get(key, 0)
    name = f"'{CONFIG_SECTION}.{LOCALITY_KEY}.{key}'"
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{name} must be an integer, got {raw!r}")
    if raw < 0:
        raise ValueError(f"{name} must be >= 0, got {raw}")
    return raw


def _groups_by_key(keys: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    """Positions of each key, keys in first-appearance order."""
    groups: dict[Hashable, list[int]] = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    return groups


def shard_window_order(shard_keys: Sequence[Hashable], window_shards: int, seed: int) -> list[int]:
    """A permutation of ``range(len(shard_keys))`` that reads ``window_shards`` shards at a time.

    ``shard_keys[i]`` names the shard holding sample ``i``. The shards are shuffled,
    grouped into windows of ``window_shards`` and the samples of each window are
    shuffled. The same ``seed`` always gives the same order.
    """
    if window_shards <= 0:
        raise ValueError(f"window_shards must be positive, got {window_shards}")
    rng = random.Random(seed)
    groups = _groups_by_key(shard_keys)
    shards = list(groups)
    rng.shuffle(shards)
    order: list[int] = []
    for start in range(0, len(shards), window_shards):
        window = [i for shard in shards[start : start + window_shards] for i in groups[shard]]
        rng.shuffle(window)
        order.extend(window)
    return order


@dataclass(frozen=True)
class ShardWindow:
    """Shard of every sample plus the window width of :func:`shard_window_order`."""

    keys: tuple[Hashable, ...]
    shards: int

    def __post_init__(self) -> None:
        if self.shards <= 0:
            raise ValueError(f"window shards must be positive, got {self.shards}")

    def order(self, seed: int) -> list[int]:
        """The shard-window permutation of the samples for ``seed``."""
        return shard_window_order(self.keys, self.shards, seed)


def _blocks(shard_keys: Sequence[Hashable], block_size: int) -> list[list[int]]:
    """Contiguous runs of at most ``block_size`` positions that never cross a shard."""
    blocks: list[list[int]] = []
    for positions in _groups_by_key(shard_keys).values():
        blocks.extend(
            positions[start : start + block_size] for start in range(0, len(positions), block_size)
        )
    return blocks


def block_fraction_subset(
    samples: Sequence[T], shard_keys: Sequence[Hashable], block_size: int, fraction: float
) -> list[T]:
    """Shuffled blocks of ``samples`` covering ``fraction`` of them (at least one sample).

    The last block taken is cut to the exact count. The block order depends only on the
    sample list, so a larger ``fraction`` returns a superset of a smaller one.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if len(shard_keys) != len(samples):
        raise ValueError(f"{len(shard_keys)} shard keys for {len(samples)} samples")
    target = max(1, int(len(samples) * float(fraction))) if samples else 0
    blocks = _blocks(shard_keys, block_size)
    random.Random(FRACTION_SEED).shuffle(blocks)
    chosen = [samples[i] for block in blocks for i in block]
    return chosen[:target]
