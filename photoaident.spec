# photoaident.spec
import sys
from pathlib import Path

import PySide6 as _pyside6
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect insightface model data and native libs
insightface_datas = collect_data_files("insightface")
insightface_libs = collect_dynamic_libs("insightface")

# Qt geoservices plugins are not auto-discovered by PyInstaller.
# libqtgeoservices_osm.so is required for the OSM map plugin to work.
_pyside6_dir = Path(_pyside6.__file__).parent
_geoservices_src = _pyside6_dir / "Qt" / "plugins" / "geoservices"

# Collect Qt geoservices plugins in a platform-aware way so that the OSM
# plugin is bundled on Linux (.so), Windows (.dll), and macOS (.dylib).
if sys.platform.startswith("win"):
    _geoservices_patterns = ["*.dll"]
elif sys.platform == "darwin":
    _geoservices_patterns = ["*.dylib"]
else:
    # Default to Unix-like shared libraries
    _geoservices_patterns = ["*.so*"]

geoservices_binaries = []
for _pattern in _geoservices_patterns:
    for _plugin in _geoservices_src.glob(_pattern):
        if _plugin.is_file():
            geoservices_binaries.append(
                (str(_plugin), "PySide6/Qt/plugins/geoservices")
            )
a = Analysis(
    ["src/photoaident/__main__.py"],
    pathex=["src"],
    binaries=[*insightface_libs, *geoservices_binaries],
    datas=[
        *insightface_datas,
        ("assets/", "assets/"),  # icons etc.
        # Alembic migration scripts must be present at runtime so the migration
        # engine can load and execute them.  They are not auto-discovered by
        # PyInstaller because the migrations/ directory has no __init__.py.
        (
            "src/photoaident/db/migrations",
            "photoaident/db/migrations",
        ),
        # QML source for the map dialog — not a Python file, so PyInstaller
        # won't discover it automatically.  Destination mirrors the package
        # layout so that Path(__file__).parent / "map_view.qml" resolves
        # correctly when the app is frozen.
        (
            "src/photoaident/ui/widgets/map_view.qml",
            "photoaident/ui/widgets/",
        ),
    ],
    hiddenimports=[
        "insightface",
        "insightface.app",
        "insightface.model_zoo",
        "onnxruntime",
        "onnxruntime.capi._pybind_state",
        "faiss",
        "sqlalchemy.dialects.sqlite",
        "PySide6.QtDBus",  # used by _reveal_in_file_manager on Linux
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # keep as folder first (more debuggable)
    name="photoaident",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # no terminal window
    icon="assets/icons/app.png",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="photoaident",
)
