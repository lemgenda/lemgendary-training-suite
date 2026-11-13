"""Git refs the generated notebook checks out (suite and env-manager clones).

The clone cell fetches one ref per repository (a branch, a tag or a full commit SHA) and
resets the checkout to it. The default is ``main`` for both, so a plain launch clones the
default branch exactly as before; a test launch may pin the suite (and the env-manager) to
a fix branch or a commit.

A ref reaches the notebook as a Python literal and git as one argument of an argument list
(never a shell), and it is validated here first: only the characters of ordinary branch,
tag and SHA names, no leading ``-`` (git would read an option), none of git's forbidden ref
sequences (``..``, ``@{``, ``//``, a trailing ``/`` or ``.lock``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_GIT_REF = "main"
"""Ref the clone cell checks out when none is given (the repositories' default branch)."""
MAX_GIT_REF_LENGTH = 200

_REF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_FORBIDDEN_SEQUENCES = ("..", "//", "@{", "/.")
_FORBIDDEN_ENDINGS = ("/", ".", ".lock")


class GitRefError(ValueError):
    """A git ref is empty, too long or not a plain branch, tag or commit name."""


def validate_git_ref(ref: str) -> str:
    """Return ``ref`` when it is a safe branch, tag or SHA name; raise :class:`GitRefError`."""
    if not ref or len(ref) > MAX_GIT_REF_LENGTH:
        raise GitRefError(f"Git ref must have 1 to {MAX_GIT_REF_LENGTH} characters")
    if not _REF_PATTERN.fullmatch(ref):
        raise GitRefError(
            f"Git ref {ref!r} may only use letters, digits, '.', '_', '/' and '-' and must "
            "start with a letter or digit"
        )
    if any(seq in ref for seq in _FORBIDDEN_SEQUENCES) or ref.endswith(_FORBIDDEN_ENDINGS):
        raise GitRefError(f"Git ref {ref!r} is not a valid branch, tag or commit name")
    return ref


@dataclass(frozen=True)
class GitRefs:
    """Refs of the suite clone and the env-manager clone."""

    suite: str = DEFAULT_GIT_REF
    env_manager: str | None = None
    """Env-manager ref; None uses :attr:`suite` (one shared ref for both clones)."""

    def __post_init__(self) -> None:
        validate_git_ref(self.suite)
        if self.env_manager is not None:
            validate_git_ref(self.env_manager)

    @property
    def env_manager_ref(self) -> str:
        """The ref the env-manager clone checks out."""
        return self.suite if self.env_manager is None else self.env_manager

    @classmethod
    def of(cls, suite: str | None = None, env_manager: str | None = None) -> GitRefs:
        """Refs from optional option values (None keeps :data:`DEFAULT_GIT_REF`)."""
        return cls(suite=DEFAULT_GIT_REF if suite is None else suite, env_manager=env_manager)


DEFAULT_GIT_REFS = GitRefs()
