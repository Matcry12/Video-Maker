"""Video Maker — load .env at package import time."""
import os
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    try:
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    except Exception:
        pass
