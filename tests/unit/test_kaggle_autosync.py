"""Per-epoch Kaggle auto-sync: isolated payload, registry slug, throttle, merged editors.

kagglehub and the Kaggle IAM client are fakes; nothing reaches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training.cloud import kaggle_api_sync
from training.cloud.kaggle_api_sync import KaggleApiSyncService
from training.cloud.kaggle_hub import KaggleHubManager
from training.cloud.manager import CloudSyncError
from training.cloud.sync_policy import KaggleSyncPolicy
from training.training import engine

_MODEL = "nima_aesthetic_mobile"
_SLUG = "nima-aesthetic-mobile"


class _UploadRecorder:
    """Fake kagglehub that records each upload's handle and file list."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, list[str], str]] = []

    def model_upload(self, handle: str, local_model_dir: str, version_notes: str) -> None:
        """Record the uploaded handle, files (relative, POSIX) and notes."""
        root = Path(local_model_dir)
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        self.uploads.append((handle, files, version_notes))

    def upload_count(self) -> int:
        """Number of uploads."""
        return len(self.uploads)


@pytest.fixture(name="model_dir")
def _model_dir(fake_workspace: Path) -> Path:
    """A polluted model folder like the one the audit found in v28-v36."""
    root = fake_workspace / "LemGendaryModels" / _MODEL
    for rel in (
        "checkpoints/latest.pth",
        "checkpoints/best.pth",
        "checkpoints/stray.pt",
        "notebooks/train.ipynb",
        "dist/kaggle/pytorch/Model.pt",
        "history.csv",
        "metrics.csv",
        "README.md",
        "Model.onnx",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel, encoding="utf-8")
    return root


@pytest.fixture(name="recorder")
def _recorder(monkeypatch: pytest.MonkeyPatch) -> _UploadRecorder:
    recorder = _UploadRecorder()
    monkeypatch.setitem(sys.modules, "kagglehub", recorder)
    return recorder


@pytest.fixture(autouse=True)
def _restore_kaggle_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the credentials the managers export (setenv records the original state)."""
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)


def test_auto_sync_uploads_only_isolated_payload(
    model_dir: Path, recorder: _UploadRecorder
) -> None:
    """Only checkpoints/*.pth, README.md and metrics.csv go to <owner>/<registry slug>."""
    manager = KaggleHubManager(username="owner", key="KGAT_fake")
    assert manager.sync(_MODEL, 6, model_dir)

    handle, files, notes = recorder.uploads[0]
    assert handle == f"owner/{_SLUG}/pytorch/default"
    assert files == ["README.md", "checkpoints/best.pth", "checkpoints/latest.pth", "metrics.csv"]
    assert "epoch 6" in notes
    assert (model_dir / "notebooks" / "train.ipynb").is_file()
    assert not [p for p in model_dir.parent.iterdir() if p.name.startswith(".kaggle-push-")]


def test_auto_sync_failure_raises(fake_workspace: Path, recorder: _UploadRecorder) -> None:
    """A model folder without checkpoints is a CloudSyncError, not a silent upload."""
    empty = fake_workspace / "LemGendaryModels" / _MODEL
    empty.mkdir(parents=True)
    with pytest.raises(CloudSyncError):
        KaggleHubManager(username="owner", key="KGAT_fake").sync(_MODEL, 1, empty)
    assert recorder.upload_count() == 0


class TestSyncPolicy:
    """``fleet.kaggle_sync_every_epochs`` / ``fleet.kaggle_sync_on_best``."""

    def test_defaults(self) -> None:
        """Every 5 epochs and on a new best."""
        policy = KaggleSyncPolicy.from_config({})
        assert [e for e in range(1, 11) if policy.should_sync(e, False)] == [5, 10]
        assert policy.should_sync(3, True)

    def test_configured(self) -> None:
        """A cadence of 0 with best-only off never uploads per epoch."""
        policy = KaggleSyncPolicy.from_config(
            {"fleet": {"kaggle_sync_every_epochs": 0, "kaggle_sync_on_best": False}}
        )
        assert not any(policy.should_sync(e, True) for e in range(1, 20))

    def test_invalid(self) -> None:
        """Negative or wrongly typed values are rejected."""
        with pytest.raises(ValueError):
            KaggleSyncPolicy.from_config({"fleet": {"kaggle_sync_every_epochs": -1}})
        with pytest.raises(TypeError):
            KaggleSyncPolicy.from_config({"fleet": {"kaggle_sync_on_best": "yes"}})


def test_engine_throttles_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    """The training engine triggers the auto-sync only on epochs the policy allows."""
    triggered: list[int] = []
    monkeypatch.setattr(
        engine, "trigger_cloud_sync", lambda model, epoch, config: triggered.append(epoch)
    )
    ctx: Any = SimpleNamespace(
        model_name=_MODEL,
        run=SimpleNamespace(
            args=SimpleNamespace(auto_sync=True, env="kaggle"),
            config={"fleet": {"kaggle_sync_every_epochs": 3, "kaggle_sync_on_best": True}},
        ),
    )
    for epoch in range(1, 8):
        engine.maybe_trigger_cloud_sync(ctx, epoch, epoch == 2)
    assert triggered == [2, 3, 6]


def _user(name: str) -> Any:
    return SimpleNamespace(user=SimpleNamespace(user_name=name))


class _FakeIamService:
    """Holds one policy; records every write."""

    def __init__(self, members: list[Any]) -> None:
        binding = SimpleNamespace(role="editor", members=members)
        self.policy = SimpleNamespace(bindings=[SimpleNamespace(role="viewer"), binding])
        self.writes: list[Any] = []

    def get_iam_policy(self, _request: Any) -> Any:
        """Return the stored policy."""
        return self.policy

    def set_iam_policy(self, request: Any) -> None:
        """Record the written policy."""
        self.writes.append(request.policy)


_IAM = SimpleNamespace(
    CanonicalRole=SimpleNamespace(CANONICAL_ROLE_EDITOR="editor"),
    GetIamPolicyRequest=SimpleNamespace,
    SetIamPolicyRequest=SimpleNamespace,
    KaggleResourceId=SimpleNamespace,
    IamPolicyBinding=SimpleNamespace,
    Principal=SimpleNamespace,
    UserPrincipal=SimpleNamespace,
)
_SECURITY = SimpleNamespace(KaggleResourceType=SimpleNamespace(KAGGLE_RESOURCE_TYPE_MODELS="m"))


def _editor_names(service: _FakeIamService) -> list[str]:
    binding = service.policy.bindings[1]
    return [m.user.user_name for m in binding.members]


def test_editors_are_merged_not_replaced() -> None:
    """Configured collaborators (fixture) are added; existing editors stay."""
    service = _FakeIamService([_user("existing-editor"), _user("fixture-editor-a")])
    client = SimpleNamespace(security=SimpleNamespace(iam_client=service))
    merged = KaggleApiSyncService.set_editor_policy(
        client, (_IAM, _SECURITY), 7, ["fixture-editor-a", "fixture-editor-b"]
    )
    assert merged == ["existing-editor", "fixture-editor-a", "fixture-editor-b"]
    assert _editor_names(service) == merged
    assert len(service.writes) == 1
    assert service.policy.bindings[0].role == "viewer"


def test_no_write_when_all_editors_present() -> None:
    """Nothing is written when every collaborator is already an editor."""
    service = _FakeIamService([_user("fixture-editor-a")])
    client = SimpleNamespace(security=SimpleNamespace(iam_client=service))
    KaggleApiSyncService.set_editor_policy(client, (_IAM, _SECURITY), 7, ["fixture-editor-a"])
    assert not service.writes


def test_collaborators_come_from_ecosystem(monkeypatch: pytest.MonkeyPatch) -> None:
    """The editor list is ``kaggle.collaborators`` of the ecosystem file (fixture)."""
    seen: list[list[str]] = []

    def _record(_client: Any, _modules: Any, _model_id: Any, editors: list[str]) -> list[str]:
        seen.append(editors)
        return editors

    monkeypatch.setattr(kaggle_api_sync, "load_kaggle_iam", lambda: (_IAM, _SECURITY))
    monkeypatch.setattr(KaggleApiSyncService, "set_editor_policy", staticmethod(_record))
    api: Any = SimpleNamespace(
        model_get=lambda ref: SimpleNamespace(id=7),
        build_kaggle_client=_NullContext,
    )
    service = KaggleApiSyncService(username="owner", key="KGAT_fake")
    results: dict[str, Any] = {}
    service.sync_collaborators(api, "owner/model", results)
    assert seen == [["fixture-editor-a", "fixture-editor-b"]]
    assert results["collaborators"] == ["fixture-editor-a", "fixture-editor-b"]


class _NullContext:
    """``with api.build_kaggle_client() as client`` stand-in."""

    def __enter__(self) -> Any:
        return SimpleNamespace()

    def __exit__(self, *_exc: object) -> None:
        return None
