from pathlib import Path
import platform
import re
from typing import Optional

def normalize_path(p: str) -> Optional[Path]:
    """Normalize a path so it works on both Windows and Linux/WSL."""
    if not p:
        return None

    # Detect Windows-style path like C:\Users\...
    win_path_pattern = re.compile(r"^[A-Za-z]:\\")
    if win_path_pattern.match(p):
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")

    # Otherwise just return a normal Path (Linux or already normalized)
    return Path(p).expanduser().resolve()
