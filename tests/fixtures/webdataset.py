"""Shared WebDataset test fixtures: encoded images and tar shards."""

import io
import json
import tarfile
from pathlib import Path

from PIL import Image


def webp_bytes(color, size: tuple[int, int] = (32, 32), mode: str = "RGB", **save_kwargs) -> bytes:
    """Encode a solid-colour image as WebP (``save_kwargs`` go to ``Image.save``)."""
    buf = io.BytesIO()
    Image.new(mode, size, color=color).save(buf, format="WEBP", **save_kwargs)
    return buf.getvalue()


def json_bytes(payload: dict) -> bytes:
    """Encode ``payload`` as UTF-8 JSON."""
    return json.dumps(payload).encode("utf-8")


def write_tar(path: Path, members: dict[str, bytes]) -> None:
    """Write ``members`` (name -> payload, in order) as an uncompressed tar shard."""
    with tarfile.open(path, "w") as tf:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
