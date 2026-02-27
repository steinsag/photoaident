# photoaident.spec
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect insightface model data and native libs
insightface_datas = collect_data_files("insightface")
insightface_libs = collect_dynamic_libs("insightface")

a = Analysis(
    ["src/photoaident/__main__.py"],
    pathex=["src"],
    binaries=insightface_libs,
    datas=[
        *insightface_datas,
        ("assets/", "assets/"),          # icons etc.
        # Alembic migration scripts must be present at runtime so the migration
        # engine can load and execute them.  They are not auto-discovered by
        # PyInstaller because the migrations/ directory has no __init__.py.
        (
            "src/photoaident/db/migrations",
            "photoaident/db/migrations",
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
    exclude_binaries=True,      # keep as folder first (more debuggable)
    name="photoaident",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,              # no terminal window
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
