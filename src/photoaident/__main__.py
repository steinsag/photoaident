import sys

from PySide6 import QtWidgets

from photoaident.app import MyWidget
from photoaident.db.migrate import apply_migrations
from photoaident.paths import AppPaths


def main():
    # Initialize paths and database
    paths = AppPaths()
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")

    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
