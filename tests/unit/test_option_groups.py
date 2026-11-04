"""Unit tests for ``training.cli.option_groups.option_group``."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import dataclass
from typing import Annotated

import typer
from typer.testing import CliRunner

from training.cli.option_groups import option_group


@dataclass(frozen=True)
class _Group:
    """Two options, one required."""

    name: Annotated[str, typer.Option("--name")]
    count: Annotated[int, typer.Option("--count")] = 1


@dataclass(frozen=True)
class _Extra:
    """A second group for stacking."""

    loud: Annotated[bool, typer.Option("--loud")] = False


class TestOptionGroup(unittest.TestCase):
    """The decorator exposes dataclass fields as keyword parameters."""

    def test_group_fields_replace_the_group_parameter(self) -> None:
        """Typer sees the group fields where the group parameter was, in order."""

        @option_group("opts", _Group)
        def command(target: str, *, opts: _Group, tail: bool = False) -> None:
            typer.echo(f"{target} {opts} {tail}")

        params = inspect.signature(command).parameters
        self.assertEqual(list(params), ["target", "name", "count", "tail"])
        self.assertIs(params["name"].default, inspect.Parameter.empty)
        self.assertEqual(params["count"].default, 1)

    def test_typer_parses_stacked_groups(self) -> None:
        """A Typer command receives each group as one dataclass instance."""
        app = typer.Typer()

        @app.command()
        @option_group("opts", _Group)
        @option_group("extra", _Extra)
        def command(target: str, *, opts: _Group, extra: _Extra) -> None:
            typer.echo(f"{target}|{opts.name}|{opts.count}|{extra.loud}")

        result = CliRunner().invoke(app, ["t", "--name", "n", "--count", "3", "--loud"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.strip(), "t|n|3|True")

        missing = CliRunner().invoke(app, ["t"])
        self.assertNotEqual(missing.exit_code, 0)

    def test_missing_group_parameter_is_rejected(self) -> None:
        """A command without the named parameter cannot be decorated."""

        def command(target: str) -> str:
            return target

        with self.assertRaises(TypeError):
            option_group("opts", _Group)(command)

    def test_clashing_parameter_is_rejected(self) -> None:
        """A command parameter may not share a name with a group field."""

        def command(name: str, *, opts: _Group) -> tuple[str, _Group]:
            return name, opts

        with self.assertRaises(TypeError):
            option_group("opts", _Group)(command)


if __name__ == "__main__":
    unittest.main()
