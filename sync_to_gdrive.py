"""Backward compatibility shim for sync_to_gdrive.py.

Delegates directly to tools.sync_to_gdrive.
"""

from tools.sync_to_gdrive import main

if __name__ == "__main__":
    main()
