"""Resume refusal record: the trainer writes it, the Kaggle job reports its text verbatim.

Every Kaggle call is a fake; nothing is pushed and nothing runs on Kaggle.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest

from tests.unit.test_kaggle_kernel_bundle import _FakeKaggle, _RecordingJobs
from training.core_loop import resume_recording_refusal
from training.run_settings.policy import FrozenSettingsChanged
from training.run_settings.record import SettingChange
from training.run_settings.refusal import RESUME_REFUSAL_RECORD, ResumeRefusalRecord
from training.server import job_kaggle_run

REFUSED = FrozenSettingsChanged({"model.kwargs": SettingChange({"width": 8}, {"width": 16})})


def _refuse() -> None:
    raise REFUSED


def test_record_keeps_the_refusal_text(tmp_path: Path) -> None:
    """The record holds exactly the exception text; clear removes it."""
    record = ResumeRefusalRecord.in_folder(tmp_path)
    assert record.read() is None
    record.write(REFUSED)
    assert record.read() == str(REFUSED)
    record.clear()
    assert not record.path.exists()


def test_unreadable_record_reads_as_none(tmp_path: Path) -> None:
    """A malformed record or one without error text yields no refusal."""
    record = ResumeRefusalRecord.in_folder(tmp_path)
    record.path.write_text("{not json", encoding="utf-8")
    assert record.read() is None
    record.path.write_text('{"settings": []}', encoding="utf-8")
    assert record.read() is None


def test_trainer_records_refusal_and_clears_stale_one(tmp_path: Path) -> None:
    """A refused resume writes the record and re-raises; a resume that passes clears it."""
    record = ResumeRefusalRecord.in_folder(tmp_path)
    with pytest.raises(FrozenSettingsChanged):
        resume_recording_refusal(_refuse, record)
    assert record.read() == str(REFUSED)
    assert resume_recording_refusal(lambda: "state", record) == "state"
    assert not record.path.exists()


class _RefusingKaggle(_FakeKaggle):
    """Every session ends with ``statuses`` and a resume refusal record in its output."""

    def kernels_output(self, _handle: str, path: str, file_pattern: str) -> None:
        """Write the payload plus the refusal record."""
        super().kernels_output(_handle, path, file_pattern)
        ResumeRefusalRecord.in_folder(Path(path) / "LemGendaryModels" / "codeformer").write(REFUSED)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api: _FakeKaggle) -> _RecordingJobs:
    monkeypatch.setattr(job_kaggle_run, "resolve_kaggle_credentials", lambda: ("u", "KGAT_x"))
    monkeypatch.setattr(job_kaggle_run, "authenticated_kaggle_api", lambda _u, _k: api)
    jobs = _RecordingJobs(tmp_path / "hub")
    event = threading.Event()
    monkeypatch.setattr(event, "wait", lambda _s=None: False)
    jobs.execute_kaggle_training("job", "codeformer", {"auto_pull": False}, event)
    return jobs


@pytest.mark.parametrize("status", ["COMPLETE", "ERROR"])
def test_kaggle_job_error_is_the_refusal_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """A refused resume fails the Kaggle job with the local job's error text."""
    api = _RefusingKaggle((status,))
    jobs = _run(tmp_path, monkeypatch, api)
    assert jobs.statuses[-1]["status"] == "failed"
    assert jobs.statuses[-1]["error_message"] == str(REFUSED)
    assert re.escape(RESUME_REFUSAL_RECORD) in api.output_pattern
    assert len(api.pushed) == 1


def test_kaggle_failure_without_refusal_keeps_status_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a refusal record a failed kernel reports its Kaggle status."""
    jobs = _run(tmp_path, monkeypatch, _FakeKaggle(("ERROR",)))
    assert jobs.statuses[-1]["error_message"] == "Kaggle kernel ended with status ERROR"
