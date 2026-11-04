"""``training_status`` / ``is_fully_trained`` on every GUI model entry.

Covers the verdict function, the ratchet pair rule for the data-fraction
pillar, each status reached through ``GET /api/gui/models/with-stats``, the
``unknown`` entry for a malformed registry model (the whole list used to fail
with HTTP 500), and the OpenAPI schema that marks both fields as required.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from training.server.app import create_app
from training.server.routes.gui_status import RungTracker, training_verdict

LADDER = [256, 384, 512]
CSV_HEADER = "Epoch,Res,Data,PSNR\n"


def _model(**extra: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "name": "Test Model",
        "category": "restoration",
        "class_name": "TestNet",
        "optimization": {"res_ladder": LADDER},
        "sota_targets": {"psnr": 30.0},
    }
    info.update(extra)
    return info


@pytest.fixture(name="project")
def project_fixture(tmp_path: Path) -> Path:
    """Empty project root with a small registry."""
    root = tmp_path / "project"
    root.mkdir()
    registry = {
        "models": {
            "fresh_model": _model(),
            "weights_model": _model(),
            "partial_model": _model(),
            "full_model": _model(parameters_m=12.5),
            "stale_fraction_model": _model(),
            "broken_model": _model(sota_targets=["psnr"], epochs="many"),
        }
    }
    (root / "unified_models_v2.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir()
    (ckpt_root / "weights_model.pt").write_bytes(b"weights")
    _write_csv(ckpt_root, "partial_model", [(1, 256, 0.25, 20.0), (2, 256, 0.5, 22.0)])
    _write_csv(
        ckpt_root,
        "full_model",
        [(e, 256 if e < 10 else 512, 1.0, 31.0) for e in range(1, 21)],
    )
    # Full data at 256 px, then the step up to 512 px with no fraction reported:
    # under the pair rule the fraction resets, so the data pillar is not passed.
    rows = [(e, 256, 1.0, 31.0) for e in range(1, 20)] + [(20, 512, None, 31.0)]
    _write_csv(ckpt_root, "stale_fraction_model", rows)
    return root


def _write_csv(ckpt_root: Path, key: str, rows: list[tuple[int, int, float | None, float]]) -> None:
    folder = ckpt_root / key
    folder.mkdir()
    lines = [CSV_HEADER]
    for epoch, res, frac, psnr in rows:
        lines.append(f"{epoch},{res},{'' if frac is None else frac},{psnr}\n")
    (folder / "metrics.csv").write_text("".join(lines), encoding="utf-8")


@pytest.fixture(name="client")
def client_fixture(project: Path) -> Iterator[TestClient]:
    """Test client for an app serving the fixture project."""
    app = create_app(project_root=project, enforce_auth=False)
    with TestClient(app) as client:
        yield client
    app.state.job_manager.shutdown()


@pytest.fixture(name="entries")
def entries_fixture(client: TestClient) -> dict[str, dict[str, Any]]:
    """``with-stats`` entries keyed by model key."""
    resp = client.get("/api/gui/models/with-stats")
    assert resp.status_code == 200, resp.text
    return {e["model_key"]: e for e in resp.json()}


def test_every_entry_carries_both_fields(entries: dict[str, dict[str, Any]]) -> None:
    """All registry models are listed, each with a status and a verdict."""
    assert len(entries) == 6
    for entry in entries.values():
        assert entry["training_status"] in {
            "fully_trained",
            "partially_trained",
            "weights_ready",
            "initializing",
            "unknown",
        }
        assert isinstance(entry["is_fully_trained"], bool)


@pytest.mark.parametrize(
    ("key", "status", "fully"),
    [
        ("fresh_model", "initializing", False),
        ("weights_model", "weights_ready", False),
        ("partial_model", "partially_trained", False),
        ("full_model", "fully_trained", True),
        ("stale_fraction_model", "partially_trained", False),
    ],
)
def test_status_paths(
    entries: dict[str, dict[str, Any]], key: str, status: str, fully: bool
) -> None:
    """Each training state is reached and agrees with ``is_fully_trained``."""
    entry = entries[key]
    assert entry["training_status"] == status
    assert entry["is_fully_trained"] is fully
    assert entry["stats_error"] is None


def test_stale_fraction_does_not_pass_data_pillar(entries: dict[str, dict[str, Any]]) -> None:
    """A full fraction at a lower resolution step does not count at the top rung."""
    entry = entries["stale_fraction_model"]
    assert entry["ladder_passed"] is True
    assert entry["data_fraction_passed"] is False
    assert entry["data_fraction_completed"] == 0.0


def test_unmeasured_metrics_are_not_reported_as_achieved(
    entries: dict[str, dict[str, Any]],
) -> None:
    """Review D-08: a checkpoint without metrics shows no value and no pass for its targets."""
    entry = entries["weights_model"]
    assert entry["best_metric"] is None
    assert entry["sota_target"] == 30.0
    psnr = next(d for d in entry["best_metrics_details"] if d["key"] == "psnr")
    assert psnr["value"] is None
    assert psnr["passed"] is False


def test_parameter_count_comes_from_registry(entries: dict[str, dict[str, Any]]) -> None:
    """Review D-08: ``parameters_m`` is the registry value, or null when the registry has none."""
    assert entries["full_model"]["parameters_m"] == 12.5
    assert entries["fresh_model"]["parameters_m"] is None


def test_malformed_registry_entry_is_unknown(entries: dict[str, dict[str, Any]]) -> None:
    """A model whose stats fail is still listed, with status ``unknown``."""
    entry = entries["broken_model"]
    assert entry["training_status"] == "unknown"
    assert entry["is_fully_trained"] is False
    assert entry["stats_error"]


def test_openapi_marks_fields_required(client: TestClient) -> None:
    """The live schema documents both fields as required on every entry."""
    schema = client.get("/openapi.json").json()
    op = schema["paths"]["/api/gui/models/with-stats"]["get"]
    items = op["responses"]["200"]["content"]["application/json"]["schema"]["items"]
    name = items["$ref"].rsplit("/", 1)[-1]
    model = schema["components"]["schemas"][name]
    assert {"training_status", "is_fully_trained"} <= set(model["required"])
    assert model["properties"]["is_fully_trained"]["type"] == "boolean"
    assert "unknown" in model["properties"]["training_status"]["enum"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"epochs_completed": 0, "has_checkpoint": False}, ("initializing", False)),
        ({"epochs_completed": 0, "has_checkpoint": True}, ("weights_ready", False)),
        ({"epochs_completed": 5, "has_checkpoint": True}, ("partially_trained", False)),
        ({"epochs_completed": 19, "has_checkpoint": True}, ("partially_trained", False)),
        ({"epochs_completed": 20, "has_checkpoint": True}, ("fully_trained", True)),
    ],
)
def test_training_verdict(kwargs: dict[str, Any], expected: tuple[str, bool]) -> None:
    """The verdict needs 20 epochs plus the other 3 pillars."""
    verdict = training_verdict(
        sota_reached=True, ladder_passed=True, data_fraction_passed=True, **kwargs
    )
    assert tuple(verdict) == expected


@pytest.mark.parametrize("pillar", ["sota_reached", "ladder_passed", "data_fraction_passed"])
def test_training_verdict_needs_every_pillar(pillar: str) -> None:
    """A single failed pillar keeps the model partially trained."""
    pillars = {"sota_reached": True, "ladder_passed": True, "data_fraction_passed": True}
    pillars[pillar] = False
    verdict = training_verdict(epochs_completed=30, has_checkpoint=True, **pillars)
    assert tuple(verdict) == ("partially_trained", False)


def test_rung_tracker_pair_rule() -> None:
    """The fraction carries within a step, resets on a step up, and must reach full data."""
    tracker = RungTracker()
    assert not tracker.full_data_at_top(LADDER)
    tracker.add(256, 1.0)
    tracker.add(None, None)
    assert tracker.current == (256, 1.0)
    tracker.add(512, None)
    assert tracker.current == (512, 0.0)
    assert not tracker.full_data_at_top(LADDER)
    tracker.add(None, 1.0)
    assert tracker.full_data_at_top(LADDER)


def test_rung_tracker_logs_ratchet_violation(caplog: pytest.LogCaptureFixture) -> None:
    """A fraction that drops within one step is logged; the last row still wins."""
    tracker = RungTracker("m.csv")
    tracker.add(512, 1.0)
    tracker.add(512, 0.5)
    assert tracker.current == (512, 0.5)
    assert "Ratchet violation in m.csv" in caplog.text
