"""Exception tuples for recoverable failure modes that several modules share.

Members are listed in alphabetical order.

Single source of truth: a handler that recovers from one of these failure modes
catches the tuple defined here, so every call site agrees on what "recoverable"
means. Anything outside a tuple is a bug and propagates.
"""

from __future__ import annotations

ErrorTypes = tuple[type[BaseException], ...]

# Restoring optimizer, scheduler or governor state, or rebuilding datasets and loaders
# after a governor audit: bad state values, missing attributes, CUDA or I/O errors.
STATE_RESTORE_ERRORS: ErrorTypes = (
    AttributeError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

# Starting a cloud sync: manager construction, thread start, filesystem.
CLOUD_SYNC_START_ERRORS: ErrorTypes = (OSError, RuntimeError, ValueError, TypeError, KeyError)

# A caller-supplied epoch callback: metric formatting and log-broadcast I/O.
EPOCH_CALLBACK_ERRORS: ErrorTypes = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

# Constructing torchmetrics modules: missing backends, weight download or cache, device move.
METRIC_INIT_ERRORS: ErrorTypes = (ImportError, OSError, RuntimeError, ValueError, TypeError)
