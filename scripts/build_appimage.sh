#!/usr/bin/env bash
set -e

# First build the PyInstaller bundle
#./scripts/build.sh

echo "→ Preparing AppDir..."
mkdir -p AppDir/usr/bin
cp dist/photoaident/photoaident AppDir/usr/bin/
cp -r dist/photoaident/_internal AppDir/usr/bin/

mkdir -p AppDir/usr/share/icons/hicolor/256x256/apps/
cp assets/icons/app.png AppDir/usr/share/icons/hicolor/256x256/apps/photoaident.png

echo "→ Building AppImage via Docker..."
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)":/workspace:Z \
    -w /workspace \
    appimagecrafters/appimage-builder:latest \
    appimage-builder --recipe AppImageBuilder.yml

echo "→ Done! PhotoAIdent-0.1.0-x86_64.AppImage ready."