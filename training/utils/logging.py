"""Logging subsystem and ForceTTY stream wrapper for LemGendary Model Training Suite."""

import logging
import sys
from pathlib import Path
from typing import Any, TextIO


class ForceTTY:
    """Stream wrapper forcing isatty() to True for unbuffered terminal and notebook progress bars."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def write(self, data: str) -> int:
        return self.stream.write(data)

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return True

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.stream, attr)


def install_force_tty() -> None:
    """Wrap sys.stdout and sys.stderr with ForceTTY if not already wrapped."""
    if not isinstance(sys.stdout, ForceTTY):
        sys.stdout = ForceTTY(sys.stdout)
    if not isinstance(sys.stderr, ForceTTY):
        sys.stderr = ForceTTY(sys.stderr)


def setup_logging(
    log_file: Path | str | None = None,
    verbose: bool = False,
    logger_name: str = "lemtrain",
) -> logging.Logger:
    """Configure structured logger with console stream and optional file logging."""
    logger = logging.getLogger(logger_name)
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Avoid duplicate handlers on re-configuration
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        file_path = Path(log_file).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
