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

COMMANDS: list[list[str]] = [
    ["ruff", "check", "--fix", "."],
    ["black", "."],
    ["ty", "check"],
    ["pytest"],
]


def check_translations() -> int:
    """Check if translation source files (*.ts) are up-to-date."""
    print("\n[verify] checking translations (i18n)...\n", flush=True)

    ts_files = [
        "assets/translations/photoaident_de.ts",
        "assets/translations/photoaident_en.ts",
    ]

    # In a real scenario, we might want to discover these files automatically
    # or use a project file. For now, we list them explicitly.

    try:
        # Run lupdate to see if any strings are missing or obsolete
        subprocess.run(
            [
                "pyside6-lupdate",
                "src/photoaident/app.py",
                "src/photoaident/ui/preferences_dialog.py",
                "-ts",
            ]
            + ts_files,
            check=True,
            capture_output=True,
            text=True,
        )

        # Check if git detects changes in .ts files
        result = subprocess.run(
            ["git", "diff", "--exit-code"] + ts_files,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("[verify] Translations are out of date! Please update them with:")
            print(
                "  uv run pyside6-lupdate src/photoaident/app.py "
                "src/photoaident/ui/preferences_dialog.py -ts " + " ".join(ts_files)
            )
            return result.returncode

    except subprocess.CalledProcessError as e:
        print(f"[verify] Translation check failed: {e}")
        if e.stderr:
            print(e.stderr)
        return e.returncode

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
    for tool in ("ruff", "black", "ty", "pytest", "pyside6-lupdate", "git"):
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
