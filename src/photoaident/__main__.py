import sys

from PySide6 import QtWidgets

from photoaident.app import MainWindow
from photoaident.db.migrate import apply_migrations
from photoaident.paths import AppPaths
from photoaident.utils.instance_lock import InstanceLock


def main():
    # Initialize paths and database
    paths = AppPaths()
    paths.ensure_dirs()

    # Try to acquire instance lock
    lock = InstanceLock(paths.lock_path)
    if not lock.acquire():
        # Another instance is already running
        app = QtWidgets.QApplication([])
        from photoaident.app import load_translations

        load_translations(app)
        QtWidgets.QMessageBox.critical(
            None,
            "PhotoAIdent",
            QtWidgets.QApplication.translate(
                "InstanceLock",
                "Another instance of PhotoAIdent is already running.\n"
                "Please close it before starting a new one.",
            ),
        )
        sys.exit(1)

    apply_migrations(f"sqlite:///{paths.db_path}")

    app = QtWidgets.QApplication([])

    from photoaident.app import load_translations

    load_translations(app)

    window = MainWindow(paths)
    window.show()

    try:
        sys.exit(app.exec())
    finally:
        lock.release()


if __name__ == "__main__":
    main()
