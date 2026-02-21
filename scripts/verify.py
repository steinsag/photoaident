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


def _ensure_available(cmd: str) -> None:
    if shutil.which(cmd) is None:
        print(
            f"verify: required tool '{cmd}' not found on PATH.\n"
            "Hint: run 'uv sync' to install dev dependencies."
        )
        sys.exit(127)


def run() -> int:
    # Ensure required tools are present before running
    for tool in ("ruff", "black", "ty", "pytest"):
        _ensure_available(tool)

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
