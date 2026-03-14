"""Helpers to persist and restore Qt widget geometry via QSettings (INI format)."""

from pathlib import Path

from PySide6 import QtCore, QtWidgets


def save_widget_geometry(
    widget: QtWidgets.QWidget,
    ini_path: Path,
    *,
    save_state: bool = False,
) -> None:
    """Save widget geometry (and optionally main-window state) to an INI file.

    Args:
        widget: The widget whose geometry to save.
        ini_path: Path to the INI file used for persistence.
        save_state: If True and widget is a QMainWindow, also persist
            ``saveState()`` (toolbar/dock positions).
    """
    settings = QtCore.QSettings(str(ini_path), QtCore.QSettings.Format.IniFormat)
    settings.beginGroup(type(widget).__name__)
    settings.setValue("geometry", widget.saveGeometry())
    if save_state and isinstance(widget, QtWidgets.QMainWindow):
        settings.setValue("state", widget.saveState())
    settings.endGroup()


def restore_widget_geometry(
    widget: QtWidgets.QWidget,
    ini_path: Path,
    *,
    restore_state: bool = False,
) -> bool:
    """Restore widget geometry (and optionally main-window state) from an INI file.

    Args:
        widget: The widget whose geometry to restore.
        ini_path: Path to the INI file used for persistence.
        restore_state: If True and widget is a QMainWindow, also restore
            ``restoreState()`` (toolbar/dock positions).

    Returns:
        True if saved geometry was found and applied, False otherwise.
    """
    settings = QtCore.QSettings(str(ini_path), QtCore.QSettings.Format.IniFormat)
    settings.beginGroup(type(widget).__name__)
    geometry = settings.value("geometry")
    if geometry is None:
        settings.endGroup()
        return False

    widget.restoreGeometry(geometry)

    # Guard against the window being restored off-screen (e.g. after an
    # external monitor is disconnected).  frameGeometry() reflects the stored
    # position even before the widget is shown, so we can check it here.
    frame = widget.frameGeometry()
    on_any_screen = any(
        screen.availableGeometry().intersects(frame)
        for screen in QtWidgets.QApplication.screens()
    )
    if not on_any_screen:
        settings.endGroup()
        return False

    if restore_state and isinstance(widget, QtWidgets.QMainWindow):
        state = settings.value("state")
        if state:
            widget.restoreState(state)
    settings.endGroup()
    return True
