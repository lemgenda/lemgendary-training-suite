"""Backward compatibility shim for judicial_audit_api.py.

Delegates directly to tools.judicial_audit.
"""

from tools.judicial_audit import (
    SimpleDataset,
    load_pytorch_model,
    main,
    run_judicial_audit,
)

if __name__ == "__main__":
    main()
