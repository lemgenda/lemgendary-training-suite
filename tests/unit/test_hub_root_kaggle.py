"""The models hub resolves from one place, also in the Kaggle layout (R3-K3 item 1).

On Kaggle the suite is cloned to ``/kaggle/temp`` while ``LemGendaryModels`` must live under
``/kaggle/working`` (the kernel output). The notebook exports ``LEMGENDARY_WORKSPACE_ROOT``; these
tests fake that layout with temp folders (the workspace root is not the suite's parent folder)
and check that training, export, recovery, sync and soup all use the same hub.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training import platform_paths as platform_paths_module
from training.checkpoint.recovery import CheckpointRecoveryEngine
from training.cloud.git_hub import GitHubSyncManager
from training.core_loop import build_training_paths
from training.data.manifold import ManifoldResolver
from training.export.common import resolve_export_paths
from training.export.forex_exporter import ForexModelExporter
from training.model_soup import resolve_models_root
from training.platform_paths import (
    KAGGLE_RUN_TYPE_ENV,
    KAGGLE_WORKING_DIR_ENV,
    PlatformPaths,
    on_kaggle,
    platform_paths,
)
from training.utils.paths import MODELS_FOLDER, get_models_hub_root, get_project_root

MODEL = "mirnet_exposure"


@pytest.fixture(name="working_hub")
def fixture_working_hub(fake_workspace: Path) -> Path:
    """The hub under the fake working dir (``LEMGENDARY_WORKSPACE_ROOT``)."""
    assert fake_workspace.resolve() != get_project_root().parent.resolve()
    return (fake_workspace / MODELS_FOLDER).resolve()


def test_hub_root_follows_workspace_root_not_project_parent(working_hub: Path) -> None:
    """With the canonical project root the hub is under the workspace root."""
    assert get_models_hub_root() == working_hub
    assert get_models_hub_root(get_project_root()) == working_hub


def test_hub_root_ignores_kaggle_working_dir_env(
    working_hub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``KAGGLE_WORKING_DIR`` alone no longer moves the hub (one source: the workspace root)."""
    monkeypatch.setenv(KAGGLE_WORKING_DIR_ENV, str(working_hub.parent / "elsewhere"))
    assert get_models_hub_root() == working_hub


def test_training_paths_land_in_the_working_hub(working_hub: Path) -> None:
    """``build_training_paths`` writes checkpoints into the kernel-output hub."""
    paths = build_training_paths(get_project_root(), MODEL)
    assert paths.local_checkpoint_dir == working_hub / MODEL / "checkpoints"
    assert paths.local_checkpoint_dir.is_dir()
    assert paths.models_hub is not None
    assert paths.models_hub.metrics_csv == working_hub / MODEL / "metrics.csv"


def test_exporters_write_into_the_working_hub(working_hub: Path) -> None:
    """Both export destination resolvers use the same hub."""
    assert ForexModelExporter().resolve_destination(MODEL) == working_hub / MODEL
    _, production_dir = resolve_export_paths(MODEL, {"filename": "MirNet"})
    assert production_dir == working_hub / MODEL


def test_git_sync_and_soup_use_the_working_hub(working_hub: Path) -> None:
    """Auto-sync (GitHub target) and the model soup read the same hub."""
    assert GitHubSyncManager(pat="token-for-test").hub_root == working_hub
    assert resolve_models_root(get_project_root()) == working_hub


def test_recovery_reads_the_working_hub(working_hub: Path) -> None:
    """Recovery finds checkpoints in the hub the training writes to."""
    ckpt_dir = working_hub / MODEL / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    roots = CheckpointRecoveryEngine().find_candidate_roots(MODEL)
    assert ckpt_dir in roots


def test_recovery_uses_platform_kaggle_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Kaggle the recovery engine reads the working hub and input dir from platform paths."""
    fake = PlatformPaths(
        working_dir=str(tmp_path / "working"),
        checkout_dir=str(tmp_path / "temp"),
        input_dir=str(tmp_path / "input"),
    )
    monkeypatch.setattr("training.checkpoint.recovery.kaggle_paths", lambda: fake)
    monkeypatch.setattr("training.checkpoint.recovery._probe_kagglehub_checkpoints", _no_probe)
    hub_ckpt = Path(fake.hub_root) / MODEL / "checkpoints"
    hub_ckpt.mkdir(parents=True)
    mount = Path(fake.input_dir) / "owner" / f"{MODEL}-checkpoints"
    mount.mkdir(parents=True)
    roots = CheckpointRecoveryEngine(env="kaggle").find_candidate_roots(MODEL)
    assert hub_ckpt.resolve() in roots
    assert mount.resolve() in roots


def _no_probe(_kh_mod: object, _model_name: str) -> None:
    """KaggleHub probe stand-in: no network in tests."""


def test_on_kaggle_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kaggle detection reads the run-type variable Kaggle sets in every kernel."""
    monkeypatch.delenv(KAGGLE_RUN_TYPE_ENV, raising=False)
    monkeypatch.delenv(KAGGLE_WORKING_DIR_ENV, raising=False)
    monkeypatch.setattr(
        platform_paths_module,
        "_KAGGLE",
        PlatformPaths(working_dir="missing-working-dir", checkout_dir="missing"),
    )
    assert not on_kaggle()
    monkeypatch.setenv(KAGGLE_RUN_TYPE_ENV, "Interactive")
    assert on_kaggle()


def test_platform_dataset_roots_feed_the_manifold_resolver() -> None:
    """The manifold resolver searches the platform dataset roots on Kaggle and Colab."""
    for env in ("kaggle", "colab"):
        roots = ManifoldResolver(env=env).get_search_roots()
        for root in platform_paths(env).dataset_roots:
            assert Path(root).resolve() in roots


def test_kaggle_hub_root_is_under_the_working_dir() -> None:
    """The rendered notebook hub (working dir) matches what the resolver uses there."""
    kaggle = platform_paths("kaggle")
    assert kaggle.hub_root == f"{kaggle.working_dir}/{MODELS_FOLDER}"
    assert not kaggle.hub_root.startswith(kaggle.checkout_dir)
