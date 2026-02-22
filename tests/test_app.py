from PySide6 import QtCore

from photoaident.app import MainWindow
from photoaident.paths import AppPaths
from photoaident.db.migrate import apply_migrations


def test_app_setup(qtbot, tmp_path):
    """
    Test that the application window can be instantiated.
    """
    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()

    # Apply migrations to the test database
    apply_migrations(f"sqlite:///{paths.db_path}")

    window = MainWindow(paths, check_gpu=False)
    qtbot.addWidget(window)

    assert window.windowTitle() == "PhotoAIdent"
    assert window.library_page is not None
    assert window.labelling_page is not None
