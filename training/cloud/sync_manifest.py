"""Local record of what a model folder last exchanged with the Kaggle Models Hub.

``<hub root>/<model>/.kaggle_sync.json`` remembers, per variation key:

* ``pulled_version``: the Hub version number last merged into the folder, so a watcher
  pulls again only when the Hub has a newer version (K-GUI-5);
* ``pushed_files``: SHA-256 of every file of the payload last uploaded, so a push skips a
  variation whose staged payload is byte-identical (K-MDL-5).

The file starts with a dot and is not a payload file of any variation
(:func:`training.cloud.variations.is_default_payload_file`), so it is never uploaded.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("lemtrain.cloud.sync_manifest")

MANIFEST_NAME = ".kaggle_sync.json"
_HASH_CHUNK = 1024 * 1024


def payload_hashes(payload_dir: Path) -> dict[str, str]:
    """SHA-256 of every file under ``payload_dir``, keyed by POSIX relative path."""
    hashes: dict[str, str] = {}
    for path in sorted(p for p in payload_dir.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            chunk = handle.read(_HASH_CHUNK)
            while chunk:
                digest.update(chunk)
                chunk = handle.read(_HASH_CHUNK)
        hashes[path.relative_to(payload_dir).as_posix()] = digest.hexdigest()
    return hashes


class SyncManifest:
    """Read/modify/write access to one model folder's ``.kaggle_sync.json``."""

    def __init__(self, model_dir: Path) -> None:
        self.path = model_dir / MANIFEST_NAME
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable Kaggle sync record %s: %s", self.path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _entry(self, variation: str) -> dict[str, Any]:
        entry = self._data.setdefault(variation, {})
        if not isinstance(entry, dict):
            entry = {}
            self._data[variation] = entry
        return entry

    def _get(self, variation: str) -> dict[str, Any]:
        entry = self._data.get(variation)
        return entry if isinstance(entry, dict) else {}

    def save(self) -> None:
        """Write the record (creating the model folder when needed)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def pulled_version(self, variation: str) -> int | None:
        """Hub version last merged for ``variation``, or None."""
        value = self._get(variation).get("pulled_version")
        return int(value) if isinstance(value, int) else None

    def record_pull(self, variation: str, version: int) -> None:
        """Remember that Hub ``version`` of ``variation`` was merged (call :meth:`save`)."""
        self._entry(variation)["pulled_version"] = int(version)

    def pushed_hashes(self, variation: str) -> dict[str, str]:
        """File hashes of the payload last uploaded for ``variation`` (empty when none)."""
        value = self._get(variation).get("pushed_files")
        return dict(value) if isinstance(value, dict) else {}

    def record_push(self, variation: str, hashes: dict[str, str], notes: str) -> None:
        """Remember the uploaded payload of ``variation`` (call :meth:`save`)."""
        entry = self._entry(variation)
        entry["pushed_files"] = dict(hashes)
        entry["pushed_notes"] = notes
