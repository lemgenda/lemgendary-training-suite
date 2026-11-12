"""The single task-type resolver reproduces what the three former resolvers returned.

``tests/fixtures/task_type_snapshot.json`` was captured from the run-spec
(``core_loop``), governor (``governor_setup``) and epoch-scorer (``engine``)
resolvers immediately before they were merged into
``training.registry.task_type.resolve_task_type``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from training.core_loop import load_model_info, load_workspace_config
from training.governance.quality_score import score_metrics
from training.registry.task_type import DEFAULT_TASK_TYPE, resolve_task_type
from training.utils.paths import get_project_root

_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "task_type_snapshot.json"
_SNAPSHOT: dict[str, dict[str, str]] = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
_PROJECT_ROOT = get_project_root()
_CONFIG = load_workspace_config(_PROJECT_ROOT)


def _registry_model_keys() -> list[str]:
    """Every model key of the unified registry (``_``-prefixed metadata excluded)."""
    registry_path = _PROJECT_ROOT / _CONFIG.get("unified_models", "unified_models_v2.yaml")
    registry: dict[str, Any] = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    models = registry.get("models", registry)
    return sorted(key for key in models if not key.startswith("_"))


_MODEL_KEYS = _registry_model_keys()


def test_snapshot_covers_the_whole_registry() -> None:
    """A model added to the registry must be added to the snapshot too."""
    assert sorted(_SNAPSHOT) == _MODEL_KEYS


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_resolver_matches_pre_merge_outputs(model_key: str) -> None:
    """Run spec, governor and scorer all get what they got before the merge."""
    model_info = load_model_info(_PROJECT_ROOT, _CONFIG, model_key)
    resolved = resolve_task_type(model_info)
    before = _SNAPSHOT[model_key]
    assert resolved == before["run"]
    assert resolved == before["governor"]
    if before["score"].startswith("["):
        # The old scorer stringified a list-valued ``dataset_type`` (``"['detection', ...]"``);
        # the first entry is the task, and the quality score it yields is unchanged.
        metrics = {"map50": 0.5, "map50_95": 0.3}
        targets = {"map50": 0.6, "map50_95": 0.4}
        assert score_metrics(metrics, targets, resolved) == score_metrics(
            metrics, targets, before["score"]
        )
    else:
        assert resolved == before["score"]


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [
        ({"task_type": "llm", "dataset_type": "text_instruction"}, "llm"),
        ({"dataset_type": ["detection", "pose"], "category": "yolo"}, "detection"),
        ({"task_type": [], "dataset_type": "", "category": "forex"}, "forex"),
        ({}, DEFAULT_TASK_TYPE),
    ],
)
def test_field_precedence(model_info: dict[str, Any], expected: str) -> None:
    """``task_type`` beats ``dataset_type`` beats ``category``; empty values are skipped."""
    assert resolve_task_type(model_info) == expected
