#!/usr/bin/env bash
set -e

echo "→ Building PyInstaller bundle..."
./scripts/build_pyinstaller.sh

echo "→ Preparing AppDir..."
rm -rf AppDir
mkdir -p AppDir/usr/bin
mkdir -p AppDir/usr/share/icons/hicolor/512x512/apps/
mkdir -p AppDir/usr/share/icons/hicolor/256x256/apps/
mkdir -p AppDir/usr/share/icons/hicolor/128x128/apps/
mkdir -p AppDir/usr/share/icons/hicolor/64x64/apps/
mkdir -p AppDir/usr/share/icons/hicolor/48x48/apps/
mkdir -p AppDir/usr/share/applications/

# Copy PyInstaller bundle
cp -r dist/photoaident/* AppDir/usr/bin/

# Icons
cp assets/icons/app-512.png AppDir/usr/share/icons/hicolor/512x512/apps/photoaident.png
cp assets/icons/app-256.png AppDir/usr/share/icons/hicolor/256x256/apps/photoaident.png
cp assets/icons/app-128.png AppDir/usr/share/icons/hicolor/128x128/apps/photoaident.png
cp assets/icons/app-64.png AppDir/usr/share/icons/hicolor/64x64/apps/photoaident.png
cp assets/icons/app-48.png AppDir/usr/share/icons/hicolor/48x48/apps/photoaident.png
cp assets/icons/app.png AppDir/photoaident.png  # required by appimagetool in root

# Desktop file (required by appimagetool)
cat > AppDir/usr/share/applications/photoaident.desktop << EOF
[Desktop Entry]
Type=Application
Name=PhotoAIdent
Exec=photoaident
Icon=photoaident
Categories=Graphics;Photography;
EOF

# Symlink desktop file and icon to AppDir root (appimagetool requires this)
cp AppDir/usr/share/applications/photoaident.desktop AppDir/photoaident.desktop

# AppRun - the entry point appimagetool calls
cat > AppDir/AppRun << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
exec "$HERE/usr/bin/photoaident" "$@"
EOF
chmod +x AppDir/AppRun

echo "→ Building AppImage..."
ARCH=x86_64 appimagetool AppDir PhotoAIdent-0.1.0-x86_64.AppImage

echo "→ Done! PhotoAIdent-0.1.0-x86_64.AppImage ready."
