#!/usr/bin/env bash
set -e

echo "→ Cleaning previous build..."
rm -rf dist/ build/

echo "→ Generating translations..."
for ts in assets/translations/*.ts; do
    uv run pyside6-lrelease "$ts" -qm "${ts%.ts}.qm"
done

echo "→ Running PyInstaller..."
uv run pyinstaller photoaident.spec

echo "→ Done! Output in dist/photoaident/"
echo "   Run with: ./dist/photoaident/photoaident"
