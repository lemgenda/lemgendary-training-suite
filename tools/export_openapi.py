"""Export the frozen ``openapi.json`` contract from the live sidecar FastAPI app.

Usage (from the project root)::

    python tools/export_openapi.py            # rewrite openapi.json
    python tools/export_openapi.py --check    # exit 1 if openapi.json drifted

The app is built with ``training.server.app.create_app`` exactly as the daemon
serves it (default ``enforce_auth``). Its ``ServerState`` creates a
``.lemtrain_server`` folder under the project root it is given, so the export
points it at a throw-away temporary directory instead of the real project.

Importing the app pulls in the full training stack (``torch``, ``kaggle``, ...);
run this script in an environment where those packages are installed.

The output is deterministic: the app's own key order, 2-space indent, UTF-8,
LF line endings and a trailing newline. ``tests/unit/test_openapi_contract.py``
fails when the committed file differs from this rendering. WebSocket routes
(``/api/ws/...``) are not part of OpenAPI and therefore not in the file.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENAPI_FILE = PROJECT_ROOT / "openapi.json"


def render_openapi(schema: dict[str, Any]) -> str:
    """Serialise an OpenAPI schema dict to the canonical committed text."""
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def build_schema() -> dict[str, Any]:
    """Build the OpenAPI schema from the live app object."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from training.server.app import create_app

    with tempfile.TemporaryDirectory(prefix="lemtrain-openapi-") as state_root:
        app = create_app(project_root=Path(state_root))
        schema = app.openapi()
        app.state.job_manager.executor.shutdown(wait=True)
    return schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if openapi.json differs from the app.",
    )
    args = parser.parse_args(argv)

    rendered = render_openapi(build_schema())
    if args.check:
        current = (
            OPENAPI_FILE.read_text(encoding="utf-8") if OPENAPI_FILE.exists() else ""
        )
        if current != rendered:
            print(
                "openapi.json is out of date; run: python tools/export_openapi.py",
                file=sys.stderr,
            )
            return 1
        print("openapi.json is up to date.")
        return 0

    OPENAPI_FILE.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {OPENAPI_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
