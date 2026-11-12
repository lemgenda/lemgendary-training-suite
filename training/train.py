"""Script entry point for training: fixes sys.path, then delegates to training.core_loop.main."""

import importlib.util
import os
import sys

# --- Hyper-Verbose Path Defense (2026 Specialization) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)

if script_dir in sys.path:
    sys.path.remove(script_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


def _bootstrap_venv_path() -> None:
    """Run ``training.utils.paths.bootstrap_sys_path`` before any package import.

    The module is loaded from its file because importing the ``training`` package may need
    the virtual environment's site-packages this call adds.
    """
    paths_file = os.path.join(script_dir, "utils", "paths.py")
    spec = importlib.util.spec_from_file_location("_lemtrain_bootstrap_paths", paths_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {paths_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bootstrap_sys_path()


_bootstrap_venv_path()

if __name__ == "__main__":
    # Imported only after the sys.path fix above so the workspace root resolves first.
    from training.core_loop import main

    main()
