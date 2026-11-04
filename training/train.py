"""Script entry point for training: fixes sys.path, then delegates to training.core_loop.main."""

import os
import sys

# --- Hyper-Verbose Path Defense (2026 Specialization) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
venv_site_pkgs = os.path.normpath(os.path.join(workspace_root, ".venv", "Lib", "site-packages"))

if script_dir in sys.path:
    sys.path.remove(script_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if os.path.exists(venv_site_pkgs) and venv_site_pkgs not in sys.path:
    sys.path.insert(0, venv_site_pkgs)

if __name__ == "__main__":
    # Imported only after the sys.path fix above so the workspace root resolves first.
    from training.core_loop import main

    main()
