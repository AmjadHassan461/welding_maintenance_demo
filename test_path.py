from pathlib import Path
import platform
import re

def normalize_path(p: str):
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


# Your actual Windows path
win_path = r"C:\Users\amjad\OneDrive\Desktop\Predictive Maintenance For Robotic Welding Arm\welding_maintenance_demo V2.0\Datasets\CV Seam Detection Dataset\test\images\IMG_20241119_152432_1_jpg.rf.7542ebfbd5cc31c6cf9e8f3ab24a36c7.jpg"

norm = normalize_path(win_path)

print("Original:", win_path)
print("Normalized:", norm)
print("Exists?", norm.exists() if norm is not None else "Invalid path")
