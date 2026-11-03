"""
LemGendary Model Training Suite — browser-origin guard tests.

Checks the shared security contract: only the AI Studio GUI origins may call
the sidecar from a browser, browser writes need the client header, and
non-browser clients (no Origin header) keep working.

The real app (``training.server.app``) imports the training stack (torch,
kaggle) through its routes, so these tests build a small app with the same
``install_security`` call the real app uses, and check the real wiring by
reading ``training/server/app.py`` statically. ``tests/unit/test_server.py``
exercises the guard on the real app where the full stack is installed.
"""

import ast
import unittest
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from training.server.security import (
    ALLOWED_HEADERS,
    ALLOWED_METHODS,
    ALLOWED_ORIGINS,
    AUTH_TOKEN_HEADER,
    CLIENT_HEADER,
    CLIENT_HEADER_VALUE,
    install_security,
    origin_guard_middleware,
)

GUI_ORIGIN = "http://tauri.localhost"
FOREIGN_ORIGIN = "https://evil.example"
APP_SOURCE = Path(__file__).resolve().parents[2] / "training" / "server" / "app.py"


def _build_app() -> FastAPI:
    """Small app secured exactly like training.server.app.create_app()."""
    test_app = FastAPI()

    @test_app.get("/read")
    async def read() -> dict:
        return {"ok": True}

    @test_app.post("/write")
    async def write() -> dict:
        return {"ok": True}

    @test_app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"ok": True})
        await websocket.close()

    install_security(test_app)
    return test_app


class TestOriginGuard(unittest.TestCase):
    """Behaviour of origin_guard_middleware together with CORS."""

    def setUp(self) -> None:
        self.client = TestClient(_build_app())

    def test_no_origin_read_allowed(self) -> None:
        self.assertEqual(self.client.get("/read").status_code, 200)

    def test_no_origin_write_allowed(self) -> None:
        self.assertEqual(self.client.post("/write").status_code, 200)

    def test_gui_origin_read_allowed(self) -> None:
        resp = self.client.get("/read", headers={"Origin": GUI_ORIGIN})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("access-control-allow-origin"), GUI_ORIGIN)
        self.assertIsNone(resp.headers.get("access-control-allow-credentials"))

    def test_foreign_origin_read_refused(self) -> None:
        resp = self.client.get("/read", headers={"Origin": FOREIGN_ORIGIN})
        self.assertEqual(resp.status_code, 403)
        self.assertIn("not allowed", resp.json()["detail"])

    def test_null_origin_refused(self) -> None:
        resp = self.client.get("/read", headers={"Origin": "null"})
        self.assertEqual(resp.status_code, 403)

    def test_gui_origin_write_without_header_refused(self) -> None:
        resp = self.client.post("/write", headers={"Origin": GUI_ORIGIN})
        self.assertEqual(resp.status_code, 403)
        self.assertIn(CLIENT_HEADER, resp.json()["detail"])

    def test_gui_origin_write_with_wrong_header_refused(self) -> None:
        resp = self.client.post("/write", headers={"Origin": GUI_ORIGIN, CLIENT_HEADER: "other"})
        self.assertEqual(resp.status_code, 403)

    def test_gui_origin_write_with_header_allowed(self) -> None:
        resp = self.client.post(
            "/write", headers={"Origin": GUI_ORIGIN, CLIENT_HEADER: CLIENT_HEADER_VALUE}
        )
        self.assertEqual(resp.status_code, 200)

    def test_foreign_origin_write_with_header_refused(self) -> None:
        resp = self.client.post(
            "/write",
            headers={"Origin": FOREIGN_ORIGIN, CLIENT_HEADER: CLIENT_HEADER_VALUE},
        )
        self.assertEqual(resp.status_code, 403)

    def test_preflight_from_gui_allows_client_and_token_headers(self) -> None:
        resp = self.client.options(
            "/write",
            headers={
                "Origin": GUI_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": f"{CLIENT_HEADER}, {AUTH_TOKEN_HEADER}",
            },
        )
        self.assertEqual(resp.status_code, 200)
        allowed = resp.headers.get("access-control-allow-headers", "").lower()
        self.assertIn(CLIENT_HEADER.lower(), allowed)
        self.assertIn(AUTH_TOKEN_HEADER.lower(), allowed)

    def test_preflight_from_foreign_origin_refused(self) -> None:
        resp = self.client.options(
            "/write",
            headers={
                "Origin": FOREIGN_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(resp.status_code, 403)

    def test_websocket_gui_origin_allowed(self) -> None:
        with self.client.websocket_connect("/ws", headers={"Origin": GUI_ORIGIN}) as ws:
            self.assertEqual(ws.receive_json(), {"ok": True})

    def test_websocket_no_origin_allowed(self) -> None:
        with self.client.websocket_connect("/ws") as ws:
            self.assertEqual(ws.receive_json(), {"ok": True})

    def test_websocket_foreign_origin_refused(self) -> None:
        with (
            self.assertRaises(WebSocketDisconnect),
            self.client.websocket_connect("/ws", headers={"Origin": FOREIGN_ORIGIN}) as ws,
        ):
            ws.receive_json()


class TestInstallSecurity(unittest.TestCase):
    """install_security() sets a closed CORS policy and the guard outermost."""

    def test_middleware_configuration(self) -> None:
        test_app = _build_app()
        classes = [m.cls for m in test_app.user_middleware]
        self.assertEqual(classes, [origin_guard_middleware, CORSMiddleware])

        cors = test_app.user_middleware[1]
        self.assertEqual(list(cors.kwargs["allow_origins"]), list(ALLOWED_ORIGINS))
        self.assertNotIn("*", cors.kwargs["allow_origins"])
        self.assertFalse(cors.kwargs["allow_credentials"])
        self.assertEqual(list(cors.kwargs["allow_methods"]), list(ALLOWED_METHODS))
        self.assertEqual(list(cors.kwargs["allow_headers"]), list(ALLOWED_HEADERS))
        self.assertIn(AUTH_TOKEN_HEADER, cors.kwargs["allow_headers"])
        self.assertIn(CLIENT_HEADER, cors.kwargs["allow_headers"])


class TestServerWiring(unittest.TestCase):
    """training/server/app.py uses install_security and nothing else for CORS."""

    def setUp(self) -> None:
        self.tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))

    def _calls(self) -> list[ast.Call]:
        return [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]

    @staticmethod
    def _name(func: ast.expr) -> str:
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return ""

    def test_create_app_installs_security(self) -> None:
        create_app = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "create_app"
        )
        installs = [
            call
            for call in ast.walk(create_app)
            if isinstance(call, ast.Call) and self._name(call.func) == "install_security"
        ]
        self.assertEqual(len(installs), 1)
        arg = installs[0].args[0]
        self.assertIsInstance(arg, ast.Name)
        self.assertEqual(arg.id, "application")

    def test_no_other_middleware_or_cors(self) -> None:
        names = {self._name(call.func) for call in self._calls()}
        self.assertNotIn("add_middleware", names)
        self.assertNotIn("CORSMiddleware", APP_SOURCE.read_text(encoding="utf-8"))

    def test_token_header_comes_from_security_module(self) -> None:
        key_headers = [call for call in self._calls() if self._name(call.func) == "APIKeyHeader"]
        self.assertEqual(len(key_headers), 1)
        name_kw = next(kw for kw in key_headers[0].keywords if kw.arg == "name")
        self.assertIsInstance(name_kw.value, ast.Name)
        self.assertEqual(name_kw.value.id, "AUTH_TOKEN_HEADER")


if __name__ == "__main__":
    unittest.main()
