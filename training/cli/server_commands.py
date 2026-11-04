"""lemtrain server: FastAPI sidecar daemon start, stop, status and OpenAPI export."""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import psutil
import typer
import uvicorn

from training.ecosystem import TRAINING_SERVICE, bind_host, service_port
from training.server.app import get_app
from training.server.state import ServerState

logger = logging.getLogger("lemtrain.cli.lemtrain")


server_app = typer.Typer(name="server", help="FastAPI sidecar daemon process management.")


def sidecar_port() -> int:
    """Port of the training-suite sidecar, read from the shared ecosystem file."""
    return service_port(TRAINING_SERVICE)


def is_sidecar_online(port: int | None = None) -> bool:
    """Check if the training suite daemon sidecar is listening on ``port`` (default: ecosystem)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.1)
        return s.connect_ex((bind_host(), port or sidecar_port())) == 0


def _spawn_detached_daemon(host: str, port: int) -> subprocess.Popen[bytes]:
    """Launch the uvicorn sidecar as a detached background process that outlives the CLI.

    The handle is returned rather than managed with ``with``: exiting a Popen context waits for the
    child, which would block on a daemon that is meant to keep running.
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "training.server.app:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


@server_app.command("start")
def server_start(
    host: str | None = typer.Option(
        None, "--host", "-h", help="Bind address (default: ecosystem.json bind_host)"
    ),
    port: int | None = typer.Option(
        None, "--port", "-p", help="Server port (default: ecosystem.json training-suite port)"
    ),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run detached in background"),
) -> None:
    """Start the FastAPI sidecar daemon."""
    host = host or bind_host()
    port = port or sidecar_port()
    if is_sidecar_online(port):
        typer.echo(f"[INFO] Server is already running on http://{host}:{port}")
        return

    if daemon:
        typer.echo(
            f"[START] Launching LemGendary Sidecar Daemon on http://{host}:{port} (daemon mode)..."
        )
        proc = _spawn_detached_daemon(host, port)
        state = ServerState()
        state.write_pid(proc.pid)
        token = state.get_or_create_token()
        typer.echo(f"[SUCCESS] Server daemon launched with PID {proc.pid}.")
        typer.echo(f"  Endpoint: http://{host}:{port}/api/health")
        typer.echo(f"  Interactive Docs: http://{host}:{port}/docs")
        typer.echo(f"  Auth Token: {token}")
    else:
        typer.echo(
            f"[START] Starting LemGendary Sidecar Daemon on http://{host}:{port} (foreground)..."
        )
        uvicorn.run("training.server.app:app", host=host, port=port)


@server_app.command("stop")
def server_stop() -> None:
    """Stop running sidecar daemon process."""
    state = ServerState()
    pid = state.read_pid()
    if not pid:
        typer.echo("No active server PID file found.")
        return

    try:
        if psutil.pid_exists(pid):
            p = psutil.Process(pid)
            p.terminate()
            typer.echo(f"[SUCCESS] Terminated server process with PID {pid}.")
        else:
            typer.echo(f"Process {pid} is no longer running.")
    except (psutil.Error, OSError) as e:
        logger.warning("Could not terminate sidecar process %s: %s", pid, e)
        typer.echo(f"[ERROR] Could not terminate process {pid}: {e}")
    finally:
        state.clear_pid()


@server_app.command("status")
def server_status(
    port: int | None = typer.Option(
        None, "--port", "-p", help="Target port to probe (default: ecosystem.json)"
    ),
) -> None:
    """Check sidecar daemon status and health endpoint."""
    port = port or sidecar_port()
    state = ServerState()
    pid = state.read_pid()
    online = is_sidecar_online(port)

    typer.echo("LemGendary Sidecar Daemon Telemetry:")
    typer.echo(f"  Port: {port}")
    typer.echo(f"  Listening: {'ONLINE' if online else 'OFFLINE'}")
    typer.echo(f"  PID: {pid if pid else 'None'}")
    typer.echo(f"  Token File: {state.token_path}")
    typer.echo(f"  SQLite DB: {state.db_path}")

    if online:
        try:
            url = f"http://{bind_host()}:{port}/api/health"
            req = urllib.request.Request(url, headers={"User-Agent": "LemTrain-CLI"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                typer.echo("  Health Response:")
                typer.echo(f"    Service: {data.get('service')}")
                typer.echo(f"    Version: {data.get('version')}")
                typer.echo(f"    Device: {data.get('device', {}).get('name')}")
        except (OSError, ValueError, AttributeError) as e:
            # OSError covers urllib.error.URLError/HTTPError and socket timeouts;
            # ValueError covers json.JSONDecodeError and UnicodeDecodeError.
            logger.debug("Sidecar health probe on port %s failed: %s", port, e)
            typer.echo(f"  Health probe failed: {e}")


@server_app.command("openapi")
def server_openapi(
    output: str | None = typer.Option(
        None, "--output", "-o", help="File path to save openapi.json. If omitted, prints to stdout."
    ),
) -> None:
    """Export the OpenAPI 3.1 schema specification of the sidecar daemon."""
    schema = get_app().openapi()
    content = json.dumps(schema, indent=2) + "\n"
    if output:
        out_path = Path(output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        typer.echo(f"[SUCCESS] Exported OpenAPI specification to {out_path}")
    else:
        typer.echo(content)
