# 2026: Environment Linter Sync
# pylint: disable=duplicate-code
import sys
import os

# --- Hyper-Verbose Path Defense (2026 Specialization) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
venv_site_pkgs = os.path.normpath(os.path.join(workspace_root, ".venv", "Lib", "site-packages"))

if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if os.path.exists(venv_site_pkgs) and venv_site_pkgs not in sys.path:
    sys.path.insert(0, venv_site_pkgs)

from training.core_loop import main

if __name__ == "__main__":
    main()
