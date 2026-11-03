"""Manifold task resolution: dataset_info.yaml ``task_type``, the datasets taxonomy, errors."""

import itertools
import json
import logging
import shutil
from pathlib import Path

import pytest

from training.data import task_resolution
from training.data.manifold import ManifoldResolver
from training.data.task_resolution import (
    DATASETS_TO_INTERNAL_TASK,
    TASK_TAXONOMY_ENV,
    TaskResolutionError,
    taxonomy_file,
)

TAXONOMY_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "task_taxonomy.json"

# Every way the removed folder-name heuristic told tasks apart.
MARKERS = ("parquet", "forex_dir", "forex_name", "masks", "labels", "targets")


def _old_heuristic(path: Path, name: str, container_type: str) -> str:
    """The folder-name guess ``inspect_manifold`` used before task_type/taxonomy (v16.9.99)."""
    if container_type == "parquet" or "forex" in name.lower() or (path / "forex").exists():
        return "forex"
    if (path / "masks").exists():
        return "segmentation"
    if (path / "labels").exists():
        return "face_detection"
    return "restoration"


def _manifold(root: Path, markers: tuple[str, ...], info: str | None = None) -> Path:
    name = "LemGendizedForexFixture" if "forex_name" in markers else "LemGendizedFixture"
    path = root / "-".join(markers or ("plain",)) / name
    (path / "images" / "train").mkdir(parents=True)
    for folder in ("masks", "labels", "targets"):
        if folder in markers:
            (path / folder / "train").mkdir(parents=True)
    if "forex_dir" in markers:
        (path / "forex").mkdir()
    if "parquet" in markers:
        (path / "data.parquet").touch()
    if info is not None:
        (path / "dataset_info.yaml").write_text(info, encoding="utf-8")
    return path


@pytest.fixture(name="taxonomy_env")
def _taxonomy_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(TASK_TAXONOMY_ENV, str(TAXONOMY_FIXTURE))
    task_resolution.clear_cache()
    return TAXONOMY_FIXTURE


def _inspect(path: Path) -> str:
    return ManifoldResolver(project_root=path.parent, env="local").inspect_manifold(path).task_type


@pytest.mark.parametrize(("task_id", "internal"), sorted(DATASETS_TO_INTERNAL_TASK.items()))
def test_dataset_info_task_type_maps_through_the_table(
    tmp_path: Path, task_id: str, internal: str
) -> None:
    """A declared datasets task id needs no taxonomy and maps to its internal name."""
    path = _manifold(tmp_path, ("labels",), info=f"task_type: {task_id}\n")
    assert _inspect(path) == internal


def test_dataset_info_task_type_wins_over_layout(tmp_path: Path) -> None:
    """``task_type`` beats the folder layout (a NIMA manifold with labels/ is quality)."""
    path = _manifold(tmp_path, ("masks", "labels"), info="task_type: quality\n")
    assert _inspect(path) == "quality"


def test_internal_task_name_in_dataset_info_is_kept(tmp_path: Path) -> None:
    """Manifolds that already carry an internal name keep it."""
    path = _manifold(tmp_path, (), info="task_type: face_detection\n")
    assert _inspect(path) == "face_detection"


def test_unknown_task_type_raises(tmp_path: Path) -> None:
    """A task id with no training-suite equivalent is an error, not a default."""
    path = _manifold(tmp_path, ("labels",), info="task_type: vision\n")
    with pytest.raises(TaskResolutionError, match="'vision'"):
        _inspect(path)


@pytest.mark.usefixtures("taxonomy_env")
@pytest.mark.parametrize(
    "markers",
    [combo for size in range(len(MARKERS) + 1) for combo in itertools.combinations(MARKERS, size)],
)
def test_legacy_layouts_match_the_old_heuristic(tmp_path: Path, markers: tuple[str, ...]) -> None:
    """Old-vs-new: every legacy layout (64, images only included) resolves the same way."""
    path = _manifold(tmp_path, markers)
    container = "parquet" if "parquet" in markers else "directory"
    expected = _old_heuristic(path, path.name, container)
    assert _inspect(path) == expected


@pytest.mark.usefixtures("taxonomy_env")
def test_images_only_legacy_manifold_keeps_restoration_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Images only and no task_type: published manifolds keep ``restoration``, with a warning."""
    path = _manifold(tmp_path, ())
    with caplog.at_level(logging.WARNING, logger="lemtrain.task_resolution"):
        assert _inspect(path) == "restoration"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert str(path) in message
    assert "task_type" in message


@pytest.mark.usefixtures("taxonomy_env")
def test_declared_task_type_logs_no_legacy_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The legacy default only applies when ``task_type`` is missing."""
    path = _manifold(tmp_path, (), info="task_type: segmentation\n")
    with caplog.at_level(logging.WARNING, logger="lemtrain.task_resolution"):
        assert _inspect(path) == "segmentation"
    assert not caplog.records


def test_taxonomy_found_under_the_workspace_root(tmp_path: Path, fake_workspace: Path) -> None:
    """Without the override the taxonomy is ``<workspace>/lemgendary-datasets/``."""
    target = fake_workspace / "lemgendary-datasets" / "task_taxonomy.json"
    target.parent.mkdir()
    shutil.copyfile(TAXONOMY_FIXTURE, target)
    assert taxonomy_file() == target
    assert _inspect(_manifold(tmp_path, ("masks",))) == "segmentation"


def test_missing_taxonomy_raises_naming_the_path(tmp_path: Path, fake_workspace: Path) -> None:
    """A legacy manifold with no reachable taxonomy fails with the expected location."""
    path = _manifold(tmp_path, ("labels",))
    with pytest.raises(TaskResolutionError, match="lemgendary-datasets") as err:
        _inspect(path)
    assert str(fake_workspace) in str(err.value)


def test_taxonomy_that_disagrees_with_the_legacy_layout_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the taxonomy moves ``pose`` off ``labels/`` the legacy rule is refused, not guessed."""
    raw = json.loads(TAXONOMY_FIXTURE.read_text(encoding="utf-8"))
    for task in raw["tasks"]:
        if task["id"] == "pose":
            task["layout_dir"] = None
    changed = tmp_path / "taxonomy.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv(TASK_TAXONOMY_ENV, str(changed))
    task_resolution.clear_cache()
    with pytest.raises(TaskResolutionError, match="labels"):
        _inspect(_manifold(tmp_path, ("labels",)))


def test_every_fixture_task_is_mapped_or_known_unsupported() -> None:
    """The table covers every taxonomy task the training suite can train."""
    raw = json.loads(TAXONOMY_FIXTURE.read_text(encoding="utf-8"))
    unmapped = {t["id"] for t in raw["tasks"]} - set(DATASETS_TO_INTERNAL_TASK)
    assert unmapped == {"diffusion", "vision"}
