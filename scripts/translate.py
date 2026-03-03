#!/usr/bin/env python3
"""
translate: Update Qt translation source files (*.ts) and compile them to *.qm.

Steps:
  1) pyside6-lupdate  — refresh *.ts files from Python source
  2) pyside6-lrelease — compile each *.ts → *.qm

Intended to be executed via `uv run translate` (console script) or directly
as `uv run scripts/translate.py`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

TS_FILES: list[Path] = [
    Path("assets/translations/photoaident_de.ts"),
    Path("assets/translations/photoaident_en.ts"),
]


def _ensure_available(cmd: str) -> None:
    if shutil.which(cmd) is None:
        print(
            f"translate: required tool '{cmd}' not found on PATH.\n"
            "Hint: run 'uv sync' to install dev dependencies."
        )
        sys.exit(127)


def run() -> int:
    """Run lupdate then lrelease; return 0 on success, non-zero on failure."""
    for tool in ("pyside6-lupdate", "pyside6-lrelease"):
        _ensure_available(tool)

    # Step 1: update *.ts files from Python sources
    lupdate_cmd = [
        "pyside6-lupdate",
        "-locations",
        "none",
        "-extensions",
        "py",
        "src/",
        "-ts",
        *[str(p) for p in TS_FILES],
    ]
    print(f"[translate] {' '.join(lupdate_cmd)}")
    try:
        subprocess.run(lupdate_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[translate] lupdate failed with exit code {e.returncode}")
        return e.returncode

    # Step 2: compile *.ts → *.qm
    for ts_path in TS_FILES:
        qm_path = ts_path.with_suffix(".qm")
        lrelease_cmd = ["pyside6-lrelease", str(ts_path), "-qm", str(qm_path)]
        print(f"[translate] {' '.join(lrelease_cmd)}")
        try:
            subprocess.run(lrelease_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[translate] lrelease failed with exit code {e.returncode}")
            return e.returncode

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
