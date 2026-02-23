[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=steinsag_photoaident&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=steinsag_photoaident)

# PhotoAIdent

PhotoAIdent is a local, privacy-first desktop application for AI-powered face
recognition and photo search. It scans a local photo library, detects and embeds
faces using InsightFace/ArcFace, and provides a PySide6 desktop UI where the user
progressively labels faces — assigning them to known persons or marking them as
anonymous. Over time the app learns who is in the collection and surfaces match
suggestions automatically.

No cloud, no external API calls, no data leaves the machine.

## Initial setup

This project uses package manager [uv](https://github.com/astral-sh/uv) to manage dependencies.
Run the following to install all dependencies:

    uv sync --python 3.12

## Running the app

    uv run photoaident

## Verify (format, lint, type-check, tests)

One command to auto-fix formatting/lints, run type checks, and tests:

      uv run scripts/verify.py

This runs, in order: `black .`, `ruff check --fix .`, `ty check`, `pytest`.

## Translations (i18n)

To update translation source files (`.ts`) after adding or removing strings:

    uv run pyside6-lupdate -locations none -extensions py src/ -ts assets/translations/photoaident_de.ts assets/translations/photoaident_en.ts

Flags explained:
- `-locations none` — omits `<location>` line-number tags so `.ts` files only
  change when strings actually change, not on every code reformat.  This makes
  `git diff` reliable as a staleness check.
- `-extensions py` — required when passing a directory; lupdate's default
  extension list does not include Python files.

To compile them to binary format (`.qm`) for the app to load them:

    for ts in assets/translations/*.ts; do uv run pyside6-lrelease "$ts" -qm "${ts%.ts}.qm"; done

Note: The build scripts automatically generate `.qm` files. You only need to run this manually if you want to see the
translations when running the app via `uv run photoaident`.

## Code formatting (Black)

[Black](https://black.readthedocs.io/) is used for code formatting.

Check formatting:

      uv run black --check .

Auto-format the code locally:

      uv run black .

## Linting (Ruff)

[Ruff](https://docs.astral.sh/ruff/) enforces common Python lint rules and import sorting.

Run Ruff for the whole project:

    uv run ruff check .

Optionally, to auto-fix what Ruff can fix:

    uv run ruff check . --fix

## Static type checking (ty)

[ty](https://docs.astral.sh/ty/) is used for static type checking.

Run ty for the whole project:

    uv run ty check

## Running tests (with coverage)

Pytest is configured to generate coverage reports automatically via pytest-cov.

Run tests:

    uv run pytest

After the run you'll get:

- Terminal coverage summary (missing lines shown, skip-covered enabled)
- HTML report in htmlcov/index.html
- XML report in coverage.xml

## Git pre-commit hooks

Enable the repository-managed pre-commit hook so that linters and formatters run before each commit:

    git config core.hooksPath .githooks

What it does:

- Runs ty: `uv run ty check`
- Runs Ruff lint: `uv run ruff check .`
- Runs Black in check mode: `uv run black --check .`
- Blocks the commit if linting, formatting, or typing issues are found

To bypass the hook: `git commit --no-verify`
