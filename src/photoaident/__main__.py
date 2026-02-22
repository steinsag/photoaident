import sys

from PySide6 import QtWidgets

from photoaident.app import MainWindow
from photoaident.db.migrate import apply_migrations
from photoaident.paths import AppPaths


def main():
    # Initialize paths and database
    paths = AppPaths()
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")

    app = QtWidgets.QApplication([])

    from photoaident.app import load_translations

    load_translations(app)

    window = MainWindow(paths)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
