import sys  # pragma: no cover

from PySide6 import QtWidgets  # pragma: no cover

from photoaident.app import MainWindow  # pragma: no cover
from photoaident.db.migrate import apply_migrations  # pragma: no cover
from photoaident.paths import AppPaths  # pragma: no cover
from photoaident.utils.instance_lock import InstanceLock  # pragma: no cover

APP_NAME = "PhotoAIdent"  # pragma: no cover


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
            NSString = lib.objc_getClass(b"NSString")
            return msg1_cstr(NSString, sel("stringWithUTF8String:"), s.encode())

        NSBundle = lib.objc_getClass(b"NSBundle")
        bundle = msg0(NSBundle, sel("mainBundle"))
        info = msg0(bundle, sel("infoDictionary"))
        app_ns = ns_str(APP_NAME)
        for key in ("CFBundleName", "CFBundleDisplayName"):
            msg2(info, sel("setObject:forKey:"), app_ns, ns_str(key))
    except Exception:
        pass  # Non-fatal — silently skip on unexpected environments


def main():  # pragma: no cover
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
