"""Resolution jumps of ``SmartTrainingGovernor``: rung entry fraction and ratchet."""

from __future__ import annotations

from typing import Any

import pytest

from training.governance.epoch_audit import EpochSignals
from training.governance.governor import SmartTrainingGovernor
from training.governance.governor_setup import rung_entry_fraction


def _model_info(**optimization: Any) -> dict[str, Any]:
    return {
        "name": "jump_model",
        "dataset_type": "quality",
        "input_size": 256,
        "batch_size": 8,
        "optimization": {"res_ladder": [256, 384, 512], **optimization},
    }


def _force_jump(gov: SmartTrainingGovernor) -> tuple[Any, ...]:
    """Run a SOTA-forced audit past the manifold-maturity hold."""
    gov.last_res_jump_epoch = gov.epoch_count - gov.manifold_maturity - 1
    return gov.audit_epoch(EpochSignals(current_quality=50.0), epochs_no_improve=0, force_jump=True)


def test_entry_fraction_precedence() -> None:
    """Model ``initial_fraction`` beats the config manifold default, then 0.15."""
    config = {"governor": {"manifold": {"initial_fraction": 0.3}}}
    assert rung_entry_fraction(_model_info(initial_fraction=0.2), config) == pytest.approx(0.2)
    assert rung_entry_fraction(_model_info(), config) == pytest.approx(0.3)
    assert rung_entry_fraction(_model_info(), {}) == pytest.approx(0.15)


@pytest.mark.parametrize("start_fraction", [1.0, 0.1])
def test_force_jump_enters_next_rung_at_initial_fraction(start_fraction: float) -> None:
    """A SOTA-forced jump resets the data fraction to ``initial_fraction``, not 0.5."""
    gov = SmartTrainingGovernor(_model_info(initial_fraction=0.2))
    gov.current_fraction = start_fraction
    f_changed, r_changed, *_rest, msg = _force_jump(gov)
    assert f_changed and r_changed, msg
    assert gov.current_res == 384
    assert gov.current_fraction == pytest.approx(0.2)


def test_force_jump_matches_spatial_jump_entry() -> None:
    """Forced and organic spatial jumps enter a rung at the same fraction."""
    info = _model_info(initial_fraction=0.25)
    forced = SmartTrainingGovernor(info)
    forced.current_fraction = 1.0
    _force_jump(forced)
    assert forced.current_fraction == pytest.approx(rung_entry_fraction(info, forced.config))


def test_force_jump_at_top_rung_keeps_the_rung() -> None:
    """At the last rung a forced jump changes nothing (no backward step)."""
    gov = SmartTrainingGovernor(_model_info(initial_fraction=0.2))
    gov.current_res = 512
    gov.current_fraction = 0.6
    f_changed, r_changed, *_rest, msg = _force_jump(gov)
    assert not (f_changed or r_changed)
    assert "maximum resolution" in str(msg)
    assert (gov.current_res, gov.current_fraction) == (512, pytest.approx(0.6))
