"""Unit tests for the epoch budget policy (TS D-01).

The policy is torch-free, so these tests run without the training stack.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from training.epoch_budget import (
    EpochAction,
    EpochBudgetConfig,
    EpochBudgetPolicy,
    StopReason,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _run_loop(
    policy: EpochBudgetPolicy,
    *,
    early_stop_at: int | None = None,
    converge_at: int | None = None,
    improving: bool = True,
    safety_limit: int = 1000,
) -> tuple[int, StopReason | None]:
    """Drive the policy the way the engine does and return (last epoch, reason)."""
    epoch = 1
    while epoch <= safety_limit:
        decision = policy.decide(
            epoch,
            converged=converge_at is not None and epoch >= converge_at,
            early_stop=early_stop_at is not None and epoch >= early_stop_at,
            governor_reports_improvement=improving,
        )
        if decision.action is EpochAction.STOP:
            return epoch, decision.stop_reason
        epoch += 1
    raise AssertionError("epoch loop did not terminate")


class EpochBudgetConfigTests(unittest.TestCase):
    def test_missing_section_disables_extensions(self) -> None:
        cfg = EpochBudgetConfig.from_config({})
        self.assertFalse(cfg.extensions_enabled)
        self.assertFalse(EpochBudgetConfig.from_config(None).extensions_enabled)

    def test_reads_section(self) -> None:
        cfg = EpochBudgetConfig.from_config(
            {"training_budget": {"extension_epochs": 5, "max_extension_epochs": 12}}
        )
        self.assertEqual(cfg, EpochBudgetConfig(extension_epochs=5, max_extension_epochs=12))
        self.assertTrue(cfg.extensions_enabled)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            EpochBudgetConfig.from_config({"training_budget": {"max_extension_epochs": -1}})
        for bad in ("10", 1.5, True):
            with self.subTest(value=bad), self.assertRaises(TypeError):
                EpochBudgetConfig.from_config({"training_budget": {"max_extension_epochs": bad}})
        with self.assertRaises(TypeError):
            EpochBudgetConfig.from_config({"training_budget": [1, 2]})

    def test_shipped_config_is_a_hard_cap(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = EpochBudgetConfig.from_config(config)
        self.assertEqual(cfg.max_extension_epochs, 0)
        self.assertFalse(cfg.extensions_enabled)


class EpochBudgetPolicyTests(unittest.TestCase):
    def test_rejects_empty_budget(self) -> None:
        with self.assertRaises(ValueError):
            EpochBudgetPolicy(0, EpochBudgetConfig())

    def test_one_epoch_smoke_run_stops_after_one_epoch(self) -> None:
        policy = EpochBudgetPolicy(1, EpochBudgetConfig())
        self.assertEqual(_run_loop(policy), (1, StopReason.BUDGET_REACHED))

    def test_one_epoch_run_with_shipped_config_stops_after_one_epoch(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        policy = EpochBudgetPolicy(1, EpochBudgetConfig.from_config(config))
        self.assertEqual(_run_loop(policy), (1, StopReason.BUDGET_REACHED))

    def test_budget_reached_stops(self) -> None:
        policy = EpochBudgetPolicy(7, EpochBudgetConfig())
        self.assertEqual(_run_loop(policy), (7, StopReason.BUDGET_REACHED))

    def test_continue_before_budget(self) -> None:
        policy = EpochBudgetPolicy(3, EpochBudgetConfig())
        decision = policy.decide(
            2, converged=False, early_stop=False, governor_reports_improvement=True
        )
        self.assertIs(decision.action, EpochAction.CONTINUE)
        self.assertIsNone(decision.stop_reason)

    def test_early_stop_stops_before_budget(self) -> None:
        policy = EpochBudgetPolicy(50, EpochBudgetConfig(10, 30))
        self.assertEqual(_run_loop(policy, early_stop_at=4), (4, StopReason.EARLY_STOP))

    def test_convergence_stops_and_wins_over_early_stop(self) -> None:
        policy = EpochBudgetPolicy(50, EpochBudgetConfig())
        self.assertEqual(
            _run_loop(policy, early_stop_at=20, converge_at=20), (20, StopReason.CONVERGED)
        )

    def test_extension_is_bounded_by_ceiling(self) -> None:
        policy = EpochBudgetPolicy(
            10, EpochBudgetConfig(extension_epochs=10, max_extension_epochs=25)
        )
        last_epoch, reason = _run_loop(policy, improving=True)
        self.assertEqual(last_epoch, 35)
        self.assertIs(reason, StopReason.EXTENSION_LIMIT)
        self.assertEqual(policy.extension_used, 25)

    def test_extension_reports_new_budget(self) -> None:
        policy = EpochBudgetPolicy(
            5, EpochBudgetConfig(extension_epochs=10, max_extension_epochs=30)
        )
        decision = policy.decide(
            5, converged=False, early_stop=False, governor_reports_improvement=True
        )
        self.assertIs(decision.action, EpochAction.EXTEND)
        self.assertEqual(decision.total_epochs, 15)
        self.assertEqual(decision.extension_used, 10)

    def test_no_extension_without_governor_improvement(self) -> None:
        policy = EpochBudgetPolicy(
            10, EpochBudgetConfig(extension_epochs=10, max_extension_epochs=30)
        )
        self.assertEqual(_run_loop(policy, improving=False), (10, StopReason.BUDGET_REACHED))
        self.assertEqual(policy.extension_used, 0)

    def test_early_stop_during_extension_stops(self) -> None:
        policy = EpochBudgetPolicy(
            10, EpochBudgetConfig(extension_epochs=10, max_extension_epochs=30)
        )
        self.assertEqual(_run_loop(policy, early_stop_at=14), (14, StopReason.EARLY_STOP))


if __name__ == "__main__":
    unittest.main()
