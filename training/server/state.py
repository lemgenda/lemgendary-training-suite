"""Server state, token authentication, and SQLite persistent storage subsystem.

Manages the .lemtrain_server directory, SQLite jobs database in WAL mode, PID tracking,
and token generation.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Any, Generator

from training.utils.paths import get_project_root


class ServerState:
    """Manages daemon state files, database connections, and authentication tokens."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.state_dir = self.project_root / ".lemtrain_server"
        self.logs_dir = self.state_dir / "logs"
        self.db_path = self.state_dir / "jobs.db"
        self.token_path = self.state_dir / "token"
        self.pid_path = self.state_dir / "server.pid"
        self._ensure_directories()
        self._init_db()

    def _ensure_directories(self) -> None:
        """Create state directory and logs folder."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Establish SQLite connection configured for WAL mode with deterministic closure."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize jobs table schema."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    metrics_json TEXT,
                    params_json TEXT,
                    log_file TEXT
                );
                """
            )
            conn.commit()

    def get_or_create_token(self) -> str:
        """Retrieve existing authentication token or generate a new secret."""
        if self.token_path.exists():
            token = self.token_path.read_text(encoding="utf-8").strip()
            if token:
                return token

        token = secrets.token_urlsafe(32)
        self.token_path.write_text(token, encoding="utf-8")
        return token

    def verify_token(self, provided_token: str | None) -> bool:
        """Verify provided token against local server token."""
        if not provided_token:
            return False
        expected_token = self.get_or_create_token()
        return secrets.compare_digest(provided_token.strip(), expected_token)

    def write_pid(self, pid: int | None = None) -> None:
        """Record running process PID."""
        target_pid = pid if pid is not None else os.getpid()
        self.pid_path.write_text(str(target_pid), encoding="utf-8")

    def read_pid(self) -> int | None:
        """Read recorded process PID."""
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return None

    def clear_pid(self) -> None:
        """Remove PID file on clean shutdown."""
        self.pid_path.unlink(missing_ok=True)

    def insert_job(
        self,
        job_id: str,
        job_type: str,
        model_key: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create new persistent job in pending state."""
        now = datetime.now().isoformat()
        log_file = str(self.logs_dir / f"{job_id}.log")
        params_str = json.dumps(params or {})

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, job_type, model_key, status, created_at,
                    started_at, completed_at, error_message,
                    metrics_json, params_json, log_file
                ) VALUES (?, ?, ?, 'pending', ?, NULL, NULL, NULL, '{}', ?, ?);
                """,
                (job_id, job_type, model_key, now, params_str, log_file),
            )
            conn.commit()

        return self.get_job(job_id) or {}

    def update_job_status(
        self,
        job_id: str,
        status: str,
        error_message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Update job lifecycle status and optional results."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            if status == "running":
                conn.execute(
                    "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?;",
                    (status, now, job_id),
                )
            elif status in ("completed", "failed", "cancelled"):
                metrics_str = json.dumps(metrics) if metrics is not None else None
                if metrics_str is not None:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = ?, completed_at = ?, error_message = ?, metrics_json = ?
                        WHERE id = ?;
                        """,
                        (status, now, error_message, metrics_str, job_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = ?, completed_at = ?, error_message = ?
                        WHERE id = ?;
                        """,
                        (status, now, error_message, job_id),
                    )
            else:
                conn.execute("UPDATE jobs SET status = ? WHERE id = ?;", (status, job_id))
            conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve single job record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?;", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def list_jobs(
        self,
        status: str | None = None,
        model_key: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List jobs with optional filtering."""
        query = "SELECT * FROM jobs"
        clauses: list[str] = []
        params: list[Any] = []

        if status:
            clauses.append("status = ?")
            params.append(status)
        if model_key:
            clauses.append("model_key = ?")
            params.append(model_key)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY created_at DESC LIMIT ?;"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            return [self._row_to_dict(r) for r in cursor.fetchall()]

    def recover_orphaned_jobs(self) -> int:
        """Mark unfinished jobs from prior server sessions as failed."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', completed_at = ?, error_message = 'Interrupted by server restart'
                WHERE status IN ('running', 'pending');
                """,
                (now,),
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert sqlite3.Row to deserialized dictionary."""
        d = dict(row)
        if "params_json" in d and d["params_json"]:
            try:
                d["params"] = json.loads(d["params_json"])
            except json.JSONDecodeError:
                d["params"] = {}
        else:
            d["params"] = {}

        if "metrics_json" in d and d["metrics_json"]:
            try:
                d["metrics"] = json.loads(d["metrics_json"])
            except json.JSONDecodeError:
                d["metrics"] = {}
        else:
            d["metrics"] = {}

        return d
