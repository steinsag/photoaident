import logging  # pragma: no cover
import os
import platform
import sys
from pathlib import Path

from PySide6 import QtWidgets  # pragma: no cover

from photoaident.app import MainWindow  # pragma: no cover
from photoaident.db.migrate import apply_migrations  # pragma: no cover
from photoaident.paths import AppPaths  # pragma: no cover
from photoaident.utils.instance_lock import InstanceLock  # pragma: no cover

APP_NAME = "PhotoAIdent"  # pragma: no cover


def ensure_nvidia_paths():
    if platform.system() != "Linux" or os.environ.get("ORT_PATHS_SET") == "1":
        return

    import site

    try:
        # Get the primary site-packages directory
        site_pkgs = Path(site.getsitepackages()[0])
    except IndexError:
        return

    # In CUDA/cuDNN, libs are usually in site-packages/nvidia/<pkg>/lib
    search_roots = [
        site_pkgs / "nvidia",
    ]

    valid_paths = []
    for root in search_roots:
        if not root.exists():
            continue

        # Add the root itself if it contains .so files (like tensorrt_libs)
        if any(root.glob("*.so*")):
            valid_paths.append(str(root.resolve()))

        # Also check subdirectories named 'lib' (like nvidia/cudnn/lib)
        for lib_dir in root.rglob("lib"):
            if lib_dir.is_dir():
                valid_paths.append(str(lib_dir.resolve()))

    if valid_paths:
        # Deduplicate and merge with existing LD_LIBRARY_PATH
        unique_paths = list(dict.fromkeys(valid_paths))
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(unique_paths) + (f":{current_ld}" if current_ld else "")

        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["ORT_PATHS_SET"] = "1"

        # Re-execute the script so the dynamic linker loads the new paths
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _fix_macos_app_name() -> None:  # pragma: no cover
    """Patch CFBundleName/CFBundleDisplayName via the ObjC runtime.

    Running as plain Python means macOS has no Info.plist, so the app menu
    shows "python3" / "__main__.py".  This writes the correct name into the
    live bundle dictionary before Qt builds its native menu bar.
    """
    try:
        import ctypes
        import ctypes.util

        objc_path = ctypes.util.find_library("objc")
        if objc_path is None:
            return
        lib = ctypes.cdll.LoadLibrary(objc_path)
        lib.objc_getClass.restype = ctypes.c_void_p
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        lib.objc_msgSend.restype = ctypes.c_void_p

        def sel(name: str):
            return lib.sel_registerName(name.encode())

        def msg0(obj, selector):
            lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            return lib.objc_msgSend(obj, selector)

        def msg1_cstr(obj, selector, arg: bytes):
            lib.objc_msgSend.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_char_p,
            ]
            return lib.objc_msgSend(obj, selector, arg)

        def msg2(obj, selector, arg1, arg2):
            lib.objc_msgSend.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            return lib.objc_msgSend(obj, selector, arg1, arg2)

        def ns_str(s: str):
            ns_string = lib.objc_getClass(b"NSString")
            return msg1_cstr(ns_string, sel("stringWithUTF8String:"), s.encode())

        ns_bundle = lib.objc_getClass(b"NSBundle")
        bundle = msg0(ns_bundle, sel("mainBundle"))
        info = msg0(bundle, sel("infoDictionary"))
        app_ns = ns_str(APP_NAME)
        for key in ("CFBundleName", "CFBundleDisplayName"):
            msg2(info, sel("setObject:forKey:"), app_ns, ns_str(key))
    except Exception:
        pass  # Non-fatal — silently skip on unexpected environments


def _setup_logging() -> None:  # pragma: no cover
    level = logging.DEBUG if os.environ.get("PHOTOAIDENT_DEBUG") else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():  # pragma: no cover
    _setup_logging()
    ensure_nvidia_paths()
    if sys.platform == "darwin":
        _fix_macos_app_name()

    # Initialize paths and database
    paths = AppPaths()
    paths.ensure_dirs()

    # Try to acquire instance lock
    lock = InstanceLock(paths.lock_path)
    if not lock.acquire():
        # Another instance is already running
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName(APP_NAME)
        from photoaident.app import load_translations

        load_translations(app)
        QtWidgets.QMessageBox.critical(
            None,
            APP_NAME,
            QtWidgets.QApplication.translate(
                "InstanceLock",
                "Another instance of PhotoAIdent is already running.\n"
                "Please close it before starting a new one.",
            ),
        )
        sys.exit(1)

    apply_migrations(f"sqlite:///{paths.db_path}")

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setOrganizationDomain("photoaident.app")

    from photoaident.app import load_translations

    load_translations(app)

    window = MainWindow(paths)
    window.show()

    try:
        sys.exit(app.exec())
    finally:
        lock.release()


if __name__ == "__main__":  # pragma: no cover
    main()
