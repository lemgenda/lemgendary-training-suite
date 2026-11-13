"""Generator state per DDP rank (R3-K3 item 6, K-CK-3).

A two-rank process group is simulated by patching ``torch.distributed`` (rank, world size and
``all_gather_object``), so the test runs in one process without a backend.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.distributed as dist

from training.training import step_progress
from training.utils.rng_state import (
    PAYLOAD_KEY,
    RANKS_PAYLOAD_KEY,
    capture_rng_state,
    restore_rng_state,
    rng_payload,
)


class _Group:
    """Fake two-rank group: this process plays ``rank``; the other rank's record is given."""

    def __init__(self, rank: int, other: dict[str, Any]) -> None:
        self.rank = rank
        self.other = other
        self.gathers = 0

    def all_gather_object(self, out: list[Any], obj: Any) -> None:
        """Place this rank's object at its index and the other rank's at the other index."""
        self.gathers += 1
        out[self.rank] = obj
        out[1 - self.rank] = self.other

    def get_rank(self) -> int:
        """This process's rank."""
        return self.rank


def _install(monkeypatch: pytest.MonkeyPatch, group: _Group) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda *_args, **_kwargs: group.get_rank())
    monkeypatch.setattr(dist, "all_gather_object", group.all_gather_object)


@pytest.fixture(name="two_ranks")
def _fixture_two_ranks(monkeypatch: pytest.MonkeyPatch) -> _Group:
    """Rank 0 of a two-rank group whose rank 1 has a different generator state."""
    torch.manual_seed(1)
    other = capture_rng_state()
    torch.manual_seed(0)
    group = _Group(0, other)
    _install(monkeypatch, group)
    return group


def test_all_ranks_are_saved(two_ranks: _Group) -> None:
    """The payload holds every rank's record; ``rng_state`` stays rank 0's."""
    payload = rng_payload()
    assert two_ranks.gathers == 1
    ranks = payload[RANKS_PAYLOAD_KEY]
    assert len(ranks) == 2
    assert payload[PAYLOAD_KEY] is ranks[0]
    assert not torch.equal(ranks[0]["torch_cpu"], ranks[1]["torch_cpu"])


def test_each_rank_restores_its_own_state(two_ranks: _Group) -> None:
    """Rank 1 continues from rank 1's generators, not rank 0's."""
    payload = rng_payload()
    torch.manual_seed(1)
    expected_rank1 = torch.rand(3)
    two_ranks.rank = 1
    torch.manual_seed(1234)
    assert restore_rng_state(payload)
    assert torch.equal(torch.rand(3), expected_rank1)


def test_single_rank_checkpoint_still_loads(two_ranks: _Group) -> None:
    """A checkpoint with only ``rng_state`` restores it on every rank."""
    torch.manual_seed(7)
    single = {PAYLOAD_KEY: capture_rng_state()}
    expected = torch.rand(3)
    two_ranks.rank = 1
    torch.manual_seed(99)
    assert restore_rng_state(single)
    assert torch.equal(torch.rand(3), expected)


def test_more_ranks_than_saved_fall_back_to_rank0(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank beyond the saved list restores rank 0's record."""
    torch.manual_seed(5)
    record = capture_rng_state()
    payload: dict[str, Any] = {PAYLOAD_KEY: record, RANKS_PAYLOAD_KEY: [record]}
    expected = torch.rand(2)
    group = _Group(1, {})
    _install(monkeypatch, group)
    torch.manual_seed(11)
    assert restore_rng_state(payload)
    assert torch.equal(torch.rand(2), expected)


def test_progress_saves_only_on_agreed_decision_steps(
    two_ranks: _Group, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under DDP the ranks decide together, and only on decision steps."""
    assert two_ranks.rank == 0
    reduced: list[bool] = []

    def fake_any_rank(flag: bool, _device: torch.device) -> bool:
        reduced.append(flag)
        return True

    monkeypatch.setattr(step_progress, "any_rank", fake_any_rank)
    cpu = torch.device("cpu")
    assert not step_progress.progress_save_agreed(True, 3, cpu)
    assert not reduced
    every = step_progress.DDP_DECISION_EVERY
    assert step_progress.progress_save_agreed(False, every, cpu)
    assert reduced == [False]
