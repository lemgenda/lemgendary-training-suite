"""Run setup reads its dataset root and resume anchor from ``config.yaml``.

Review D-16: the instruction dataset folder follows ``paths.datasets_root``.
Review D-15: the resume quality anchor of face and synthesis models comes from
``governor.quality_anchor_recalibration`` instead of a constant in code.
"""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from training.context_data import RunSpec, datasets_root, instruction_jsonl_paths
from training.context_resume import calibrate_quality_anchor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _spec(config: dict[str, Any], model_info: dict[str, Any] | None = None) -> RunSpec:
    return RunSpec(
        args=argparse.Namespace(),
        config=config,
        model_info=model_info or {},
        model_key="ai_helper_lemgend",
        project_root=Path("project"),
        task_type="llm",
        device="cpu",
    )


class TestDatasetsRoot(unittest.TestCase):
    """``datasets_root`` and the instruction JSONL lookup use ``paths.datasets_root``."""

    def test_relative_root_is_taken_from_project_root(self) -> None:
        """A relative ``paths.datasets_root`` is joined to the project root."""
        spec = _spec({"paths": {"datasets_root": "custom/sets"}})
        self.assertEqual(datasets_root(spec), Path("project") / "custom" / "sets")

    def test_default_root_is_the_sibling_datasets_folder(self) -> None:
        """Without the key the sibling ``LemGendaryDatasets`` folder is used."""
        self.assertEqual(datasets_root(_spec({})), Path("project") / ".." / "LemGendaryDatasets")

    def test_instruction_files_live_under_the_configured_root(self) -> None:
        """The JSONL candidates are inside ``<datasets_root>/<dataset>/data``."""
        spec = _spec({"paths": {"datasets_root": "custom"}}, {"datasets": ["MySet"]})
        train_jsonl, val_jsonl = instruction_jsonl_paths(spec)
        expected_dir = Path("project") / "custom" / "MySet" / "data"
        self.assertEqual(train_jsonl.parent, expected_dir)
        self.assertEqual(val_jsonl.parent, expected_dir)


class TestQualityAnchor(unittest.TestCase):
    """The resume anchor follows ``governor.quality_anchor_recalibration``."""

    def setUp(self) -> None:
        config_text = (PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")
        self.config: dict[str, Any] = yaml.safe_load(config_text)

    def test_shipped_limits_keep_the_previous_values(self) -> None:
        """config.yaml keeps the anchor that used to be hardcoded (600 -> 555)."""
        limits = self.config["governor"]["quality_anchor_recalibration"]
        self.assertEqual((limits["above"], limits["reset_to"]), (600.0, 555.0))

    def test_face_model_above_limit_is_reset(self) -> None:
        """A face model restored above ``above`` gets ``reset_to``."""
        governor = SimpleNamespace(best_quality=900.0)
        calibrate_quality_anchor(_spec(self.config, {"dataset_type": "face"}), governor)
        self.assertEqual(governor.best_quality, 555.0)

    def test_other_models_and_missing_section_are_unchanged(self) -> None:
        """Non-face models, and any model when the section is absent, keep their value."""
        governor = SimpleNamespace(best_quality=900.0)
        calibrate_quality_anchor(_spec(self.config, {"dataset_type": "restoration"}), governor)
        self.assertEqual(governor.best_quality, 900.0)
        calibrate_quality_anchor(_spec({}, {"dataset_type": "face"}), governor)
        self.assertEqual(governor.best_quality, 900.0)


if __name__ == "__main__":
    unittest.main()
