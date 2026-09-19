"""Backward compatibility shim for cloud_hub.py.

Delegates directly to tools.cloud_hub.
"""

import asyncio
from tools.cloud_hub import LemGendaryCloudHub, main

if __name__ == "__main__":
    asyncio.run(main())
