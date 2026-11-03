"""Canonical CLI entrypoint for LemGendary Model Training Suite."""

from training.cli.lemtrain import app


def main():
    """Run the ``lemtrain`` Typer application (same app as the ``lemtrain`` console script)."""
    app()


if __name__ == "__main__":
    main()
