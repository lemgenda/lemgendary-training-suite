"""Base utilities for notebook cell construction.

Provides standard cell dictionary builders adhering to the nbformat v4 specification.
"""

from typing import Any


def make_code_cell(source: list[str]) -> dict[str, Any]:
    """Construct a standard Jupyter nbformat v4 code cell dictionary."""
    return {
        "cell_type": "code",
        "source": source,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    }


def make_markdown_cell(source: list[str]) -> dict[str, Any]:
    """Construct a standard Jupyter nbformat v4 markdown cell dictionary."""
    return {
        "cell_type": "markdown",
        "source": source,
        "metadata": {},
    }
