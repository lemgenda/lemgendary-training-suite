"""Health probe shared by the ecosystem sidecar delegates."""

import urllib.error
import urllib.request


def sidecar_online(sidecar_url: str) -> bool:
    """Whether the sidecar at ``sidecar_url`` answers ``GET /api/health`` with HTTP 200."""
    try:
        req = urllib.request.Request(f"{sidecar_url}/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
