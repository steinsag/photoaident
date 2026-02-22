#!/usr/bin/env python3
"""
verify: Run project verification steps in order:
  1) Ruff (auto-fix)
  2) Black (format)
  3) ty (type check)
  4) pytest (tests)

Intended to be executed via `uv run verify` once exposed as a console script,
or directly as `python -m scripts.verify`.

Each step streams output and stops on first failure with the same exit code.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

COMMANDS: list[list[str]] = [
    ["ruff", "check", "--fix", "."],
    ["black", "."],
    ["ty", "check"],
    ["pytest"],
]


def check_translations() -> int:
    """Check if translation source files (*.ts) are up-to-date.

    Strategy: compare .ts file content before and after running lupdate.
    With -locations none, .ts files only change when actual translatable
    strings are added or removed — not when line numbers shift due to code
    reformatting.  A before/after comparison in Python avoids any dependency
    on the git index state and works correctly in a single run.
    """
    print("\n[verify] checking translations (i18n)...\n", flush=True)

    ts_paths = [
        Path("assets/translations/photoaident_de.ts"),
        Path("assets/translations/photoaident_en.ts"),
    ]
    update_cmd = (
        "uv run pyside6-lupdate -locations none -extensions py src/ -ts "
        + " ".join(str(p) for p in ts_paths)
    )

    before = {p: p.read_text(encoding="utf-8") for p in ts_paths}

    try:
        subprocess.run(
            # -locations none: omit <location> tags so .ts files are stable
            #   across code reformats (only change when strings change).
            # -extensions py: directory scan must be told to look at .py files
            #   (lupdate's default extension list does not include Python).
            [
                "pyside6-lupdate",
                "-locations",
                "none",
                "-extensions",
                "py",
                "src/",
                "-ts",
            ]
            + [str(p) for p in ts_paths],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[verify] lupdate failed: {e}")
        if e.stderr:
            print(e.stderr)
        return e.returncode

    after = {p: p.read_text(encoding="utf-8") for p in ts_paths}
    changed = [p for p in ts_paths if before[p] != after[p]]
    if changed:
        names = " ".join(str(p) for p in changed)
        print(
            f"[verify] Translation files were updated by lupdate: {names}\n"
            "Commit the updated files to keep translations in sync with the source:\n\n"
            f"  git add {names}\n\n"
            f"To regenerate manually:\n  {update_cmd}\n"
        )
        return 1

    print("[verify] Translations are up-to-date.")
    return 0


def _ensure_available(cmd: str) -> None:
    if shutil.which(cmd) is None:
        print(
            f"verify: required tool '{cmd}' not found on PATH.\n"
            "Hint: run 'uv sync' to install dev dependencies."
        )
        sys.exit(127)


def run() -> int:
    # Ensure required tools are present before running
    for tool in ("ruff", "black", "ty", "pytest", "pyside6-lupdate"):
        _ensure_available(tool)

    # First, check translations
    ret = check_translations()
    if ret != 0:
        return ret

    for command in COMMANDS:
        full_command = " ".join(command)
        print(f"\n[verify] running: {full_command}\n", flush=True)
        try:
            # Inherit stdout/stderr so users see the full output
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[verify] step failed with exit code {e.returncode}: {full_command}")
            return e.returncode
    print("\n[verify] all steps passed ✔")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
