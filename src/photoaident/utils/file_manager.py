import os
import subprocess
import sys
from pathlib import Path

from PySide6 import QtCore


def reveal_in_file_manager(file_path: str) -> None:
    """Reveal file_path in the system file manager, selecting it where supported.

    On macOS and Windows the file is selected directly. On Linux, the
    org.freedesktop.FileManager1 D-Bus service is tried first (selects the
    file); if unavailable, falls back to opening the parent directory via
    xdg-open.
    """
    p = Path(file_path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(p)])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(p)])
    else:  # Linux / BSD
        # Use D-Bus org.freedesktop.FileManager1 — the Linux equivalent of
        # Android intents. We call the already-running file manager directly
        # over D-Bus without spawning a subprocess, so the AppImage's
        # LD_LIBRARY_PATH never leaks into the file manager process.
        from PySide6 import QtDBus  # Linux-only module, import lazily

        file_uri = QtCore.QUrl.fromLocalFile(str(p)).toString()
        iface = QtDBus.QDBusInterface(
            "org.freedesktop.FileManager1",
            "/org/freedesktop/FileManager1",
            "org.freedesktop.FileManager1",
        )
        if iface.isValid():
            reply = iface.call("ShowItems", [file_uri], "")
            if reply.type() != QtDBus.QDBusMessage.MessageType.ErrorMessage:
                return
        # Fallback: no D-Bus file manager service, or the call was rejected.
        # Strip the AppImage library path so xdg-open's target process
        # won't pick up the bundled Qt libs.
        env = os.environ.copy()
        orig = env.pop("LD_LIBRARY_PATH_ORIG", None)
        if orig is not None:
            env["LD_LIBRARY_PATH"] = orig
        else:
            env.pop("LD_LIBRARY_PATH", None)
        subprocess.Popen(["xdg-open", str(p.parent)], env=env)
