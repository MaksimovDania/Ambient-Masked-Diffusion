from __future__ import annotations

import os
from pathlib import Path


# ==== CONFIG (edit these) =====================================================
# Folder to scan recursively:
ROOT_DIR = Path(".").resolve()

# Output file to create/overwrite:
OUTPUT_TXT = Path("python_files_dump.txt").resolve()

# Common directories to skip (by folder name):
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
}
# ==============================================================================


def iter_py_files(root: Path) -> list[Path]:
    """
    Recursively collect all *.py files under `root`, walking every subfolder.
    """
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune ignored directories so we don't descend into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]

        for name in filenames:
            if not name.endswith(".py"):
                continue
            files.append(Path(dirpath) / name)
    files.sort(key=lambda x: x.as_posix())
    return files


def main() -> None:
    root = ROOT_DIR.resolve()
    out = OUTPUT_TXT.resolve()

    py_files = iter_py_files(root)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for py_path in py_files:
            rel_path = py_path.relative_to(root).as_posix()
            f.write(rel_path + "\n\n")
            f.write("```\n")
            try:
                content = py_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = py_path.read_text(encoding="utf-8", errors="replace")
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")
            f.write("```\n\n")

    print(f"Wrote {len(py_files)} .py files to: {out}")


if __name__ == "__main__":
    main()

