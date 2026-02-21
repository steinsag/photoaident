#!/usr/bin/env bash
set -e

echo "→ Cleaning previous build..."
rm -rf dist/ build/

echo "→ Running PyInstaller..."
uv run pyinstaller photoaident.spec

echo "→ Done! Output in dist/photoaident/"
echo "   Run with: ./dist/photoaident/photoaident"
