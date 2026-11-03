"""Canonical CLI entrypoint for LemGendary Model Training Suite."""

from training.cli.lemtrain import app


def main():
    """Run the ``lemtrain`` Typer application (``[project.scripts] lemtrain = "cli:main"``)."""
    app()


if __name__ == "__main__":
    main()
