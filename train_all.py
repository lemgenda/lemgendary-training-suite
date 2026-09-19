"""Backward compatibility shim for train_all.py.

Delegates directly to tools.train_all.
"""

from tools.train_all import PHASES, main

if __name__ == "__main__":
    main()
