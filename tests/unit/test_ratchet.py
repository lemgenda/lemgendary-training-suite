"""Monotonic Ratchet (pair rule) and 4-pillar resolution tests (TS D-04).

Torch-free: the rule lives in ``training.ratchet``. The governor and the
training engine import torch, so their wiring is checked by reading the
source with ``ast`` instead of importing them.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from training.ratchet import (
    FRACTION_STATE_KEY,
    RESOLUTION_STATE_KEY,
    RatchetViolation,
    Rung,
    assert_pair_monotonic,
    check_transition,
    executed_rung,
    expand_fraction,
    next_rung_resolution,
    top_resolution_reached,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOVERNANCE_DIR = REPO_ROOT / "training" / "governance"
GOVERNOR_PATH = GOVERNANCE_DIR / "governor.py"
# Mixin modules that make up SmartTrainingGovernor; every audit-path step lives here.
AUDIT_PATHS = [
    GOVERNANCE_DIR / "epoch_audit.py",
    GOVERNANCE_DIR / "epoch_signals.py",
    GOVERNANCE_DIR / "propulsion.py",
]
GOVERNOR_PATHS = [GOVERNOR_PATH, *AUDIT_PATHS, GOVERNANCE_DIR / "governor_state.py"]
ENGINE_PATH = REPO_ROOT / "training" / "training" / "engine.py"

LADDER = [256, 384, 512]
INITIAL_FRACTION = 0.15
INCREMENT = 0.20


def _simulate_run(events: list[str]) -> list[Rung]:
    """Replay governor transitions with the same helpers the governor uses.

    Events mirror the governor paths: ``expand`` (PROPULSION), ``jump``
    (DEEPENING spatial jump, data reset), ``force`` (SOTA forced jump to
    50%), ``breakout`` (loop breaker, 15%), ``veto`` (anchor at the current
    resolution), ``hold`` (retreat/recoil: LR cooling only).
    """
    rung = Rung(LADDER[0], INITIAL_FRACTION)
    trajectory = [rung]
    for event in events:
        res, frac = rung
        nxt = next_rung_resolution(LADDER, res)
        if event == "expand":
            new = Rung(res, expand_fraction(frac, INCREMENT))
        elif event == "jump" and nxt is not None:
            new = Rung(nxt, INITIAL_FRACTION)
        elif event == "force" and nxt is not None:
            new = Rung(nxt, 0.5)
        elif event == "breakout" and nxt is not None:
            new = Rung(nxt, 0.15)
        elif event in ("veto", "hold", "jump", "force", "breakout"):
            new = Rung(res, frac)
        else:
            raise AssertionError(f"unknown event {event}")
        rung = check_transition(rung, new)
        trajectory.append(rung)
    return trajectory


class PairRuleTests(unittest.TestCase):
    """Tests for the resolution/fraction pair ratchet rule."""

    def test_simulated_run_obeys_pair_rule(self) -> None:
        """Verify a simulated event sequence never violates the pair rule."""
        events = (
            ["expand"] * 6
            + ["hold", "veto", "jump"]
            + ["expand"] * 2
            + ["hold", "force"]
            + ["expand"] * 3
            + ["breakout", "expand", "jump", "hold"]
        )
        trajectory = _simulate_run(events)

        resolutions = [r.resolution for r in trajectory]
        self.assertEqual(resolutions, sorted(resolutions), "resolution decreased")
        for res in set(resolutions):
            fractions = [r.fraction for r in trajectory if r.resolution == res]
            self.assertEqual(fractions, sorted(fractions), f"fraction decreased at {res}px")
        assert_pair_monotonic(trajectory)

        # A step-up is allowed to reset the fraction (pair rule).
        self.assertEqual(trajectory[6], Rung(256, 1.0))
        self.assertEqual(trajectory[9], Rung(384, INITIAL_FRACTION))
        self.assertEqual(trajectory[-1].resolution, 512)

    def test_resolution_decrease_is_rejected(self) -> None:
        """Verify a resolution decrease is rejected."""
        with self.assertRaises(RatchetViolation):
            check_transition(Rung(512, 0.2), Rung(384, 1.0))

    def test_fraction_decrease_within_step_is_rejected(self) -> None:
        """Verify a fraction decrease at the same resolution is rejected."""
        with self.assertRaises(RatchetViolation):
            check_transition(Rung(384, 0.75), Rung(384, 0.55))

    def test_fraction_reset_on_step_up_is_allowed(self) -> None:
        """Verify the fraction may reset when the resolution steps up."""
        self.assertEqual(check_transition(Rung(384, 1.0), Rung(512, 0.15)), Rung(512, 0.15))

    def test_assert_pair_monotonic_flags_recoil(self) -> None:
        """Verify a fraction recoil within a rung is flagged."""
        with self.assertRaises(RatchetViolation):
            assert_pair_monotonic([Rung(256, 0.15), Rung(256, 0.75), Rung(256, 0.55)])

    def test_expand_fraction_never_decreases_and_snaps(self) -> None:
        """Verify expand_fraction never decreases and snaps to 1.0."""
        self.assertAlmostEqual(expand_fraction(0.15, 0.20), 0.35)
        self.assertEqual(expand_fraction(0.95, 0.045), 1.0)
        self.assertEqual(expand_fraction(1.0, 0.20), 1.0)
        self.assertEqual(expand_fraction(0.6, -0.2), 0.6)

    def test_next_rung_resolution(self) -> None:
        """Verify next_rung_resolution returns the next ladder rung or None."""
        self.assertEqual(next_rung_resolution(LADDER, 256), 384)
        self.assertEqual(next_rung_resolution(LADDER, 300), 384)
        self.assertIsNone(next_rung_resolution(LADDER, 512))


def _methods(*paths: Path) -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found.update({n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)})
    return found


def _assigned_attrs(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


class GovernorWiringTests(unittest.TestCase):
    """Static checks that governor and engine route transitions through the ratchet."""

    def test_transitions_go_through_commit_rung(self) -> None:
        """Verify governor transition methods go through commit_rung."""
        methods = _methods(*GOVERNOR_PATHS)
        audit_steps = set(_methods(*AUDIT_PATHS))
        self.assertIn("audit_epoch", audit_steps)
        for name in sorted(audit_steps | {"recoil"}):
            with self.subTest(method=name):
                written = _assigned_attrs(methods[name])
                self.assertNotIn("current_res", written)
                self.assertNotIn("current_fraction", written)
        commit_calls = ast.dump(methods["_commit_rung"])
        self.assertIn("check_transition", commit_calls)

    def test_get_state_writes_canonical_keys(self) -> None:
        """Verify get_state writes the canonical resolution/fraction keys."""
        get_state = _methods(*GOVERNOR_PATHS)["get_state"]
        returned = next(n for n in ast.walk(get_state) if isinstance(n, ast.Return))
        assert isinstance(returned.value, ast.Dict)
        key_names = {k.id for k in returned.value.keys if isinstance(k, ast.Name)}
        self.assertIn("RESOLUTION_STATE_KEY", key_names)
        self.assertIn("FRACTION_STATE_KEY", key_names)


class CompletionResolutionTests(unittest.TestCase):
    """TS D-04: the resolution pillar reads the pre-audit ``input_size``."""

    def _ladder_passed(self, pre_audit_state: dict) -> bool:
        res, _ = executed_rung(pre_audit_state)
        return top_resolution_reached(res, LADDER)

    def test_canonical_keys(self) -> None:
        """Verify the canonical state key names."""
        self.assertEqual(RESOLUTION_STATE_KEY, "input_size")
        self.assertEqual(FRACTION_STATE_KEY, "sample_fraction")

    def test_epoch_below_top_rung_is_not_complete(self) -> None:
        """Verify an epoch below the top rung does not complete the ladder."""
        # Trained at 384 px @ 100%; the audit may then jump the governor to
        # 512 px, but the pre-audit state still says 384.
        state = {RESOLUTION_STATE_KEY: 384, FRACTION_STATE_KEY: 1.0}
        self.assertEqual(executed_rung(state), Rung(384, 1.0))
        self.assertFalse(self._ladder_passed(state))

    def test_epoch_at_top_rung_is_complete(self) -> None:
        """Verify an epoch at the top rung at 100% completes the ladder."""
        self.assertTrue(self._ladder_passed({RESOLUTION_STATE_KEY: 512, FRACTION_STATE_KEY: 1.0}))
        self.assertTrue(self._ladder_passed({RESOLUTION_STATE_KEY: [3, 512, 512]}))

    def test_legacy_or_missing_keys_never_pass(self) -> None:
        """Verify legacy or missing state keys never pass the ladder."""
        self.assertFalse(self._ladder_passed({"current_resolution": 512, "resolution": 512}))
        self.assertFalse(self._ladder_passed({}))
        self.assertEqual(executed_rung({}).fraction, 0.0)

    def test_empty_ladder_never_passes(self) -> None:
        """Verify an empty ladder never passes."""
        self.assertFalse(top_resolution_reached(2048, []))

    def test_engine_uses_pre_audit_executed_rung(self) -> None:
        """Verify the engine uses the pre-audit executed rung."""
        source = ENGINE_PATH.read_text(encoding="utf-8")
        self.assertIn("executed_rung(pre_audit_gov_state)", source)
        self.assertNotIn('"current_resolution"', source)


if __name__ == "__main__":
    unittest.main()
