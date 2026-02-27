#!/usr/bin/env bash
set -e

echo "→ Building AppImage..."

export LDAI_OUTPUT="PhotoAIdent-x86_64.AppImage"
export LINUXDEPLOY_OUTPUT_VERSION="1.0.0"

rm -rf AppDir
mkdir -p AppDir/usr/bin
mkdir -p AppDir/usr/share/metainfo
cp -r dist/photoaident/* AppDir/usr/bin/
cp assets/packaging/linux/io.github.steinsag.photoaident.appdata.xml AppDir/usr/share/metainfo/

./linuxdeploy-x86_64.AppImage --appdir AppDir -e AppDir/usr/bin/photoaident -i assets/icons/app-512.png -d assets/packaging/linux/io.github.steinsag.photoaident.desktop --output appimage

echo "→ Done! ${LDAI_OUTPUT} ready."
