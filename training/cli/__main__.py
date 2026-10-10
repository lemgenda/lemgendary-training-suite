"""Entrypoint for running training.cli directly via python -m training.cli."""

from training.cli.lemtrain import app

if __name__ == "__main__":
    app()
