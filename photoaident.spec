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
    ],
    hiddenimports=[
        "insightface",
        "insightface.app",
        "insightface.model_zoo",
        "onnxruntime",
        "onnxruntime.capi._pybind_state",
        "faiss",
        "sqlalchemy.dialects.sqlite",
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
