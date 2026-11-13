"""Kernel bundle builder, kernel status names, the watch loop and the GUI Kaggle job.

Every Kaggle call is a fake: nothing reaches the network, nothing is pushed, and the home
folder is a temporary one.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from slugify import slugify

from tests.conftest import ECOSYSTEM_FIXTURE
from training import ecosystem
from training.cloud import kernel_status
from training.cloud.hub_versions import VersionLookup
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.kernel_bundle import (
    METADATA_FILE,
    BundleRequest,
    KernelBundleError,
    build_kernel_bundle,
    checked_accelerator,
    kernel_slug,
    kernel_title,
)
from training.cloud.kernel_launch import KernelLaunchError, launch_kernel
from training.cloud.kernel_output import retrieve_kernel_output
from training.cloud.kernel_watch import KernelWatcher, WatchHooks
from training.cloud.sync_manifest import SyncManifest
from training.ecosystem import ECOSYSTEM_FILE_ENV
from training.notebooks.registry import load_registry
from training.registry.kaggle_datasets import model_kaggle_dataset_handles
from training.server import job_kaggle_run
from training.server.state import ServerState
from training.utils.lazy_imports import load_kaggle_api

_REGISTRY = load_registry()
_MODELS = sorted(k for k, v in _REGISTRY.items() if isinstance(v, dict) and not k.startswith("_"))


@pytest.fixture(autouse=True, name="isolated_home")
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake home folder and no Kaggle variables in the environment."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(home / ".kaggle"))
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME"):
        monkeypatch.delenv(var, raising=False)
    return home


def _sdk_api(requests: list[Any], monkeypatch: pytest.MonkeyPatch) -> Any:
    """The installed ``KaggleApi`` whose HTTP client records ``save_kernel`` requests."""

    def save_kernel(request: Any) -> Any:
        requests.append(request)
        return SimpleNamespace(error="", version_number=1, url="https://example.invalid/k")

    client = SimpleNamespace(
        kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(save_kernel=save_kernel))
    )
    api = load_kaggle_api()()
    monkeypatch.setattr(api, "build_kaggle_client", lambda: contextlib.nullcontext(client))
    return api


def test_registry_has_all_models() -> None:
    """The registry lists the 23 models the bundles must cover."""
    assert len(_MODELS) == 23


@pytest.mark.parametrize("model_key", _MODELS)
def test_bundle_builds_for_every_model(
    model_key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Every registry model gets a notebook bundle the installed SDK accepts unchanged."""
    request = BundleRequest(model_key, kernel_owner="pusher", resume_version=7)
    bundle = build_kernel_bundle(request, tmp_path / "bundle", _REGISTRY)

    meta = json.loads((bundle.directory / METADATA_FILE).read_text(encoding="utf-8"))
    slug = _REGISTRY[model_key]["kaggle_metadata"]["slug"]
    assert meta["id"] == f"pusher/{slug}-training"
    assert slugify(meta["title"]) == f"{slug}-training"
    assert len(f"{slug}-training") <= 50
    assert meta["kernel_type"] == "notebook" and meta["language"] == "python"
    assert meta["is_private"] is True and meta["enable_internet"] is True
    assert meta["enable_gpu"] is True and meta["enable_tpu"] is False
    assert meta["machine_shape"] == "FixtureGpuShape"
    assert "accelerator" not in meta
    assert meta["dataset_sources"] == model_kaggle_dataset_handles(_REGISTRY[model_key])
    assert meta["dataset_sources"]
    assert meta["model_sources"] == [f"fixture-owner/{slug}/pytorch/default/7"]
    notebook = json.loads((bundle.directory / meta["code_file"]).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert any("--auto_sync" in "".join(c["source"]) for c in notebook["cells"])

    sent_requests: list[Any] = []
    _sdk_api(sent_requests, monkeypatch).kernels_push(str(bundle.directory))
    sent = sent_requests[0]
    assert sent.slug == meta["id"] and sent.kernel_type == "notebook"
    assert sent.machine_shape == "FixtureGpuShape" and sent.enable_gpu is True
    assert list(sent.dataset_data_sources) == meta["dataset_sources"]
    assert list(sent.model_data_sources) == meta["model_sources"]
    assert sent.is_private is True and sent.enable_internet is True
    captured = capsys.readouterr()
    assert "does not resolve to the specified id" not in captured.out
    assert "retired" not in captured.err


def test_bundle_without_hub_version_has_no_model_sources(tmp_path: Path) -> None:
    """A model without a Hub version mounts no model source."""
    bundle = build_kernel_bundle(BundleRequest("codeformer", "pusher"), tmp_path, _REGISTRY)
    assert bundle.metadata["model_sources"] == []


def test_cpu_bundle_has_no_machine_shape(tmp_path: Path) -> None:
    """The CPU accelerator disables the GPU and sends no machine shape."""
    bundle = build_kernel_bundle(BundleRequest("codeformer", "pusher", "cpu"), tmp_path, _REGISTRY)
    assert bundle.metadata["enable_gpu"] is False
    assert "machine_shape" not in bundle.metadata


@pytest.fixture(name="retired_ecosystem")
def _retired_ecosystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """An ecosystem file whose extra accelerator uses a retired machine shape."""
    data = json.loads(ECOSYSTEM_FIXTURE.read_text(encoding="utf-8"))
    data["kaggle"]["accelerators"].append(
        {"id": "old-gpu", "label": "Old", "kind": "gpu", "machine_shape": "fixtureretiredshape"}
    )
    path = tmp_path / "ecosystem.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv(ECOSYSTEM_FILE_ENV, str(path))
    ecosystem.clear_cache()
    yield
    ecosystem.clear_cache()


@pytest.mark.usefixtures("retired_ecosystem")
def test_retired_accelerator_is_rejected() -> None:
    """A retired machine shape never reaches kernel metadata."""
    with pytest.raises(KernelBundleError, match="retired"):
        checked_accelerator("old-gpu")


def test_unknown_accelerator_is_rejected(tmp_path: Path) -> None:
    """An accelerator id missing from the ecosystem file is refused."""
    with pytest.raises(KernelBundleError, match="Unknown"):
        build_kernel_bundle(BundleRequest("codeformer", "p", "p100"), tmp_path, _REGISTRY)


def test_unknown_model_is_rejected(tmp_path: Path) -> None:
    """Only registry models (not the metadata entry) get bundles."""
    with pytest.raises(KernelBundleError, match="not found"):
        build_kernel_bundle(BundleRequest("no_such_model", "p"), tmp_path, _REGISTRY)
    with pytest.raises(KernelBundleError, match="not found"):
        build_kernel_bundle(BundleRequest("_registry_metadata", "p"), tmp_path, _REGISTRY)


def test_kernel_slug_limit() -> None:
    """Titles slugify to the slug; slugs over 50 characters are refused."""
    assert kernel_title("upn-v2") == "Upn V2 Training"
    with pytest.raises(KernelBundleError):
        kernel_slug("x" * 50)


class _FakeKaggle:
    """Fake ``KaggleApi``: records pushes, replays statuses and logs, writes kernel output."""

    def __init__(self, statuses: tuple[str, ...] = ("COMPLETE",), error: str = "") -> None:
        self.statuses = statuses
        self.logs: list[str] = []
        self.error = error
        self.pushed: list[dict[str, Any]] = []
        self.calls = 0
        self.output_pattern = ""

    def kernels_push(self, folder: str) -> Any:
        """Record the pushed metadata; answer like ``ApiSaveKernelResponse``."""
        self.pushed.append(json.loads((Path(folder) / METADATA_FILE).read_text("utf-8")))
        return SimpleNamespace(error=self.error, version_number=3, url="u")

    def kernels_status(self, _handle: str) -> Any:
        """The next scripted status (the last one repeats)."""
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        return SimpleNamespace(status=status, failure_message="")

    def kernels_logs(self, _handle: str) -> str:
        """One more log line per status poll, as a JSON stream array."""
        return json.dumps([{"data": line + "\n"} for line in self.logs[: self.calls]])

    def kernels_output(self, _handle: str, path: str, file_pattern: str) -> None:
        """Write a model payload under ``path``."""
        self.output_pattern = file_pattern
        root = Path(path) / "LemGendaryModels" / "codeformer"
        (root / "checkpoints").mkdir(parents=True)
        (root / "checkpoints" / "latest.pth").write_bytes(b"x")
        (root / "metrics.csv").write_text("epoch\n1\n", encoding="utf-8")

    def model_instance_get(self, _handle: str) -> Any:
        """No Hub instance exists."""
        raise ValueError("no such instance")


def _lookup(versions: dict[str, int | None]) -> VersionLookup:
    """Version lookup answering ``versions["latest"]`` for every handle."""
    return lambda _handle: versions["latest"]


def test_launch_resumes_latest_hub_version_and_cleans_up() -> None:
    """The launch mounts the latest pytorch/default Hub version."""
    api = _FakeKaggle()
    launch = launch_kernel(
        api, BundleRequest("ffanet_indoor", "pusher"), _lookup({"latest": 12}), _REGISTRY
    )
    assert launch.handle == "pusher/ffanet-indoor-training"
    assert launch.version_number == 3
    assert api.pushed[0]["model_sources"] == ["fixture-owner/ffanet-indoor/pytorch/default/12"]


def test_launch_reports_refusal() -> None:
    """A push answer with an error raises KernelLaunchError."""
    with pytest.raises(KernelLaunchError, match="quota"):
        launch_kernel(
            _FakeKaggle(error="quota"), BundleRequest("upn_v2", "p"), _lookup({"latest": None})
        )


def test_status_names_match_installed_sdk() -> None:
    """Status names are members of the installed KernelWorkerStatus enum."""
    enums = pytest.importorskip("kagglesdk.kernels.types.kernels_enums")
    members = set(enums.KernelWorkerStatus.__members__)
    names = {
        kernel_status.QUEUED,
        kernel_status.RUNNING,
        kernel_status.COMPLETE,
        kernel_status.ERROR,
        kernel_status.CANCEL_REQUESTED,
        kernel_status.CANCEL_ACKNOWLEDGED,
    }
    assert names <= members
    assert (
        kernel_status.status_name(
            SimpleNamespace(status=enums.KernelWorkerStatus.CANCEL_ACKNOWLEDGED)
        )
        == kernel_status.CANCEL_ACKNOWLEDGED
    )
    assert kernel_status.status_name("KernelWorkerStatus.complete") == kernel_status.COMPLETE


def test_backoff_grows_and_resets() -> None:
    """Unchanged polls back off to the maximum; a change resets."""
    backoff = kernel_status.PollBackoff(initial=30, factor=2, maximum=100)
    assert [backoff.next_interval(False) for _ in range(3)] == [60, 100, 100]
    assert backoff.next_interval(True) == 30


def test_watcher_ends_on_cancel_with_backoff() -> None:
    """A cancelled kernel ends the watch; logs are emitted once."""
    api = _FakeKaggle(("QUEUED", "RUNNING", "RUNNING", "CANCEL_ACKNOWLEDGED"))
    api.logs = ["a", "b", "c"]
    lines: list[str] = []
    sleeps: list[float] = []
    polled: list[int] = []
    hooks = WatchHooks(emit=lines.append, sleep=sleeps.append, on_poll=lambda: polled.append(1))
    result = KernelWatcher(api, "o/k", kernel_status.PollBackoff(initial=10), hooks).watch("QUEUED")
    assert result.status == kernel_status.CANCEL_ACKNOWLEDGED
    assert [line for line in lines if "STATUS" not in line] == ["a", "b", "c"]
    assert sleeps == [15.0, 10.0, 15.0]
    assert len(polled) == 2


def test_kernel_output_merges_model_payload(tmp_path: Path) -> None:
    """Kernel output payload files are merged into the hub folder."""
    api = _FakeKaggle()
    report = retrieve_kernel_output(api, "o/k", "codeformer", tmp_path / "hub").report
    assert sorted(report.written) == ["checkpoints/latest.pth", "metrics.csv"]
    assert (tmp_path / "hub" / "codeformer" / "checkpoints" / "latest.pth").is_file()
    assert "notebooks" not in api.output_pattern


def test_pull_only_when_hub_version_is_newer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pull happens only for a Hub version newer than the last pull."""
    sync = KaggleModelSyncManager(username="u", key="KGAT_fake", hub_root=tmp_path)
    pulls: list[int | None] = []
    monkeypatch.setattr(
        sync, "pull_variation", lambda _m, _v, version=None: pulls.append(version) or True
    )
    versions: dict[str, int | None] = {"latest": 4}
    assert sync.pull_default_if_newer("codeformer", _lookup(versions)) == 4
    assert sync.pull_default_if_newer("codeformer", _lookup(versions)) is None
    versions["latest"] = 5
    assert sync.pull_default_if_newer("codeformer", _lookup(versions)) == 5
    assert pulls == [4, 5]
    assert SyncManifest(tmp_path / "codeformer").pulled_version("pytorch/default") == 5


class _RecordingJobs(job_kaggle_run.KaggleRunJobsMixin):
    """The mixin with a recording job state and log sink."""

    def __init__(self, hub_root: Path) -> None:
        self.statuses: list[dict[str, Any]] = []
        self.logs: list[str] = []
        self.hub_root = hub_root
        self.project_root = hub_root
        self.state = cast(
            ServerState, SimpleNamespace(update_job_status=lambda **kw: self.statuses.append(kw))
        )
        self._broadcast_log = lambda _job, line: self.logs.append(line)

    def _hub_sync(self, user: str) -> KaggleModelSyncManager:
        """A sync manager on the test hub root."""
        return KaggleModelSyncManager(username=user, key="KGAT_fake", hub_root=self.hub_root)


def test_gui_job_pushes_notebook_bundle_and_ends_on_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GUI job pushes the notebook bundle; a cancelled kernel ends it."""
    api = _FakeKaggle(("RUNNING", "CANCEL_REQUESTED"))
    monkeypatch.setattr(job_kaggle_run, "resolve_kaggle_credentials", lambda: ("u", "KGAT_x"))
    monkeypatch.setattr(job_kaggle_run, "authenticated_kaggle_api", lambda _u, _k: api)
    jobs = _RecordingJobs(tmp_path)
    event = threading.Event()
    monkeypatch.setattr(event, "wait", lambda _s=None: False)
    jobs.execute_kaggle_training(
        "job", "mirnet_exposure", {"accelerator": None, "auto_pull": False}, event
    )
    pushed = api.pushed[0]
    assert pushed["kernel_type"] == "notebook"
    assert pushed["code_file"].endswith(".ipynb")
    assert pushed["machine_shape"] == "FixtureGpuShape"
    assert jobs.statuses[-1]["status"] == "cancelled"
