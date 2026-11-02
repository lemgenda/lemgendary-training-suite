"""
LemGendary Model Training Suite: frozen ``openapi.json`` matches the live sidecar app.

Fails when a route, schema or version changes without regenerating the
contract. Regenerate with ``python tools/export_openapi.py``.
WebSocket routes are not part of OpenAPI and are not compared.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "fastapi",
    reason="fastapi is not installed; the sidecar app cannot be built in this environment",
)

# The app imports the full training stack (torch, kaggle, onnx, ...).
try:
    from training.server.app import create_app
except ImportError as exc:
    pytest.skip(
        f"training.server.app cannot be imported in this environment: {exc}",
        allow_module_level=True,
    )

from fastapi.routing import APIRoute

from tools.export_openapi import OPENAPI_FILE, render_openapi

HTTP_METHODS = {"get", "put", "post", "delete", "patch", "head", "options", "trace"}


def _schema_operations(routes: list, prefix: str = "") -> set[tuple[str, str]]:
    """Collect (METHOD, path) for every schema route, descending into included routers.

    FastAPI >= 0.16x wraps ``include_router`` results in an included-router
    object whose routes keep their unprefixed path; the prefix lives on its
    include context. Older FastAPI flattens them into plain ``APIRoute``s.
    """
    operations: set[tuple[str, str]] = set()
    for route in routes:
        if isinstance(route, APIRoute):
            if route.include_in_schema:
                operations.update(
                    (method, prefix + route.path) for method in route.methods
                )
        elif hasattr(route, "original_router"):
            context = route.include_context
            if context.include_in_schema:
                operations |= _schema_operations(
                    route.original_router.routes, prefix + context.prefix
                )
    return operations


@pytest.fixture(scope="module")
def live_app(tmp_path_factory):
    app = create_app(project_root=tmp_path_factory.mktemp("lemtrain_state"))
    yield app
    app.state.job_manager.executor.shutdown(wait=True)


def test_openapi_json_matches_live_app(live_app) -> None:
    committed = OPENAPI_FILE.read_text(encoding="utf-8")
    assert committed == render_openapi(live_app.openapi()), (
        "openapi.json drifted from the app; run: python tools/export_openapi.py"
    )


def test_every_app_operation_is_in_openapi_json(live_app) -> None:
    committed = json.loads(OPENAPI_FILE.read_text(encoding="utf-8"))
    documented = {
        (method.upper(), path)
        for path, item in committed["paths"].items()
        for method in item
        if method in HTTP_METHODS
    }
    served = _schema_operations(live_app.routes)
    assert served, "no API routes found on the app"
    assert served == documented
