"""Typer option groups: one frozen dataclass carries a command's related options.

Typer builds a command's options from the parameters of its function, so a
command with many flags would need one parameter each. ``option_group`` lets the
command take a single dataclass instead: every field of the dataclass, annotated
``Annotated[<type>, typer.Option(...)]``, becomes a Typer option of the command,
with the same flag names, help text and defaults, and the parsed values reach
the command as one instance of that dataclass.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import typing
from collections.abc import Callable
from typing import Any


def _group_parameters(group: type) -> list[inspect.Parameter]:
    """Keyword-only parameters for the fields of ``group``, in field order."""
    hints = typing.get_type_hints(group, include_extras=True)
    params = []
    for field in dataclasses.fields(group):
        default = inspect.Parameter.empty if field.default is dataclasses.MISSING else field.default
        params.append(
            inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=hints[field.name],
            )
        )
    return params


def option_group(name: str, group: type) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Expose the fields of the dataclass ``group`` as options of a Typer command.

    The decorated command declares a keyword-only parameter ``name`` typed as
    ``group``; Typer sees one option per field in its place, and the command
    receives the parsed fields as ``group(**values)``. Apply it below
    ``@app.command(...)``; stacked decorators expose several groups.
    """
    group_params = _group_parameters(group)
    field_names = [p.name for p in group_params]

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func, eval_str=True)
        if name not in signature.parameters:
            raise TypeError(f"{func.__name__} has no parameter '{name}' for {group.__name__}")
        own = list(signature.parameters.values())
        clash = set(field_names) & {p.name for p in own}
        if clash:
            raise TypeError(f"{func.__name__} parameters clash with {group.__name__}: {clash}")
        at = list(signature.parameters).index(name)
        params = own[:at] + group_params + own[at + 1 :]

        @functools.wraps(func)
        def command(*args: Any, **kwargs: Any) -> Any:
            values = {key: kwargs.pop(key) for key in field_names if key in kwargs}
            return func(*args, **kwargs, **{name: group(**values)})

        # inspect.signature (which Typer reads) prefers an explicit __signature__.
        command.__dict__["__signature__"] = signature.replace(parameters=params)
        command.__annotations__ = {p.name: p.annotation for p in params}
        command.__annotations__["return"] = signature.return_annotation
        return command

    return decorate
