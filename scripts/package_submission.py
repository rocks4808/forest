"""
package_submission.py – Package CanopyIntel into a clean, standalone submission ZIP.

Run: python scripts/package_submission.py
Outputs: dist/CanopyIntel_Submission.zip
"""

from __future__ import annotations

import pathlib
import sys
import zipfile

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

INCLUDE_DIRS = [
    "src",
    "app",
    "tests",
    "reports",
    "models",
    "notebooks",
    "scripts",
]

INCLUDE_FILES = [
    "forest_loss_trend.csv",
    "config.yaml",
    "requirements.txt",
    "Makefile",
    "Dockerfile",
    ".dockerignore",
    "README.md",
    "EVALUATOR_GUIDE.md",
    "DESIGN.md",
    "PRODUCT.md",
    "DECISIONS.md",
    "PRD.md",
    "start.bat",
    "start.ps1",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pytest_cache",
    ".git",
    ".venv",
    ".agent",
    ".agents",
    "dist",
]


def should_exclude(path: pathlib.Path) -> bool:
    for part in path.parts:
        for pattern in EXCLUDE_PATTERNS:
            if pattern in part:
                return True
    return False


def main() -> None:
    dist_dir = ROOT_DIR / "dist"
    dist_dir.mkdir(exist_ok=True)
    zip_path = dist_dir / "CanopyIntel_Submission.zip"

    print("Packaging CanopyIntel submission bundle...")
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include specified individual files
        for filename in INCLUDE_FILES:
            file_path = ROOT_DIR / filename
            if file_path.exists():
                arcname = filename
                zf.write(file_path, arcname=arcname)
                file_count += 1

        # Include specified directories recursively
        for dirname in INCLUDE_DIRS:
            dir_path = ROOT_DIR / dirname
            if dir_path.exists():
                for item in dir_path.rglob("*"):
                    if item.is_file() and not should_exclude(item):
                        arcname = str(item.relative_to(ROOT_DIR))
                        zf.write(item, arcname=arcname)
                        file_count += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Submission bundle created successfully: {zip_path}")
    print(f"Total files packaged: {file_count}")
    print(f"Bundle size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
