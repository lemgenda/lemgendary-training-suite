"""Command line entry of the release publish: ``python -m training.cloud.release_publish_cli``.

The notebook publish cell runs this module in a fresh interpreter instead of importing
:func:`training.cloud.release_publish.publish_release` into the notebook's own kernel
process. That process may hold a torch module loaded before the dependency cell installed
another build; the export stack (torch, torchvision) must only ever load in a new process.
This module only reads the arguments, calls the publish service and prints its result.
Exit code :data:`PUBLISH_OK_EXIT_CODE` means every variation was published or unchanged.
"""

from __future__ import annotations

import argparse
import sys

from training.cloud.kaggle_model_sync import ModelPushResult
from training.cloud.release_publish import publish_release
from training.utils.logging import setup_logging

PUBLISH_OK_EXIT_CODE = 0
"""Exit code of a release whose variations were all published or unchanged."""

PUBLISH_FAILED_EXIT_CODE = 1
"""Exit code of a failed export or upload."""


def release_lines(release: ModelPushResult) -> list[str]:
    """One report line per variation, then the overall outcome."""
    lines = []
    for var_key, var_res in release.variations.items():
        if not var_res.success:
            lines.append(f"   [ERROR] {var_key}: {var_res.error}")
        elif var_res.skipped:
            lines.append(f"   [UNCHANGED] {var_key}: no new version ({var_res.handle})")
        else:
            lines.append(f"   [OK] {var_key} -> {var_res.handle}")
    if release.success:
        lines.append(
            f"[SUCCESS] Release of {release.model_key} published to the Kaggle Models Hub."
        )
    else:
        lines.append(f"[ERROR] Release publish failed: {release.errors}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Publish the release of the model named on the command line; return the exit code."""
    parser = argparse.ArgumentParser(
        description="Export a trained model and publish all its variations to Kaggle Models."
    )
    parser.add_argument("model_key", help="Model key whose hub folder is exported and pushed")
    args = parser.parse_args(argv)
    setup_logging()
    release = publish_release(args.model_key)
    for line in release_lines(release):
        print(line, flush=True)
    return PUBLISH_OK_EXIT_CODE if release.success else PUBLISH_FAILED_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
