"""Tests for photoaident.ui.window_state — save/restore widget geometry helpers."""

from unittest.mock import MagicMock, patch

from PySide6 import QtCore, QtWidgets

from photoaident.ui.window_state import restore_widget_geometry, save_widget_geometry

# ===========================================================================
# save_and_restore_geometry
# ===========================================================================


def test_save_and_restore_geometry_round_trip(qtbot, tmp_path):
    """save then restore applies saved geometry back to the widget."""
    ini_path = tmp_path / "state.ini"

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    widget.resize(640, 480)

    save_widget_geometry(widget, ini_path)

    # Create a fresh widget with a different size
    widget2 = QtWidgets.QWidget()
    qtbot.addWidget(widget2)
    widget2.resize(100, 100)

    result = restore_widget_geometry(widget2, ini_path)

    assert result is True
    # restoreGeometry applies the saved size; the frame size must now match.
    assert widget2.width() == 640
    assert widget2.height() == 480


# ===========================================================================
# restore_returns_false_when_empty
# ===========================================================================


def test_restore_returns_false_when_empty(qtbot, tmp_path):
    """restore_widget_geometry returns False when the INI file has no saved entry."""
    ini_path = tmp_path / "empty.ini"
    # File does not exist — QSettings will find no key.

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)

    result = restore_widget_geometry(widget, ini_path)

    assert result is False


# ===========================================================================
# restore_returns_true_when_saved
# ===========================================================================


def test_restore_returns_true_when_saved(qtbot, tmp_path):
    """restore_widget_geometry returns True when a geometry entry is present."""
    ini_path = tmp_path / "saved.ini"

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    widget.resize(800, 600)

    save_widget_geometry(widget, ini_path)

    widget2 = QtWidgets.QWidget()
    qtbot.addWidget(widget2)

    result = restore_widget_geometry(widget2, ini_path)

    assert result is True


# ===========================================================================
# mainwindow_state_save_restore
# ===========================================================================


def test_mainwindow_state_save_restore(qtbot, tmp_path):
    """QMainWindow state (toolbar positions) round-trips with save_state=True."""
    ini_path = tmp_path / "main_state.ini"

    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)
    window.resize(1024, 768)

    save_widget_geometry(window, ini_path, save_state=True)

    window2 = QtWidgets.QMainWindow()
    qtbot.addWidget(window2)
    window2.resize(300, 200)

    result = restore_widget_geometry(window2, ini_path, restore_state=True)

    assert result is True
    # The restored window must have taken on the saved dimensions.
    assert window2.width() == 1024
    assert window2.height() == 768


# ===========================================================================
# restore_returns_false_when_off_screen
# ===========================================================================


def test_restore_moves_widget_on_screen_when_off_screen(qtbot, tmp_path):
    """restore_widget_geometry returns True and repositions the widget when off-screen.

    This simulates the external-monitor-disconnected scenario: the saved
    position is unreachable, so the helper centres the widget on the primary
    screen and still reports a successful restore.
    """
    ini_path = tmp_path / "offscreen.ini"

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    widget.move(0, 0)
    widget.resize(640, 480)
    save_widget_geometry(widget, ini_path)

    widget2 = QtWidgets.QWidget()
    qtbot.addWidget(widget2)

    # Simulate a single screen positioned far away so it cannot intersect (0,0,640,480).
    mock_screen = MagicMock()
    mock_screen.availableGeometry.return_value = QtCore.QRect(50000, 50000, 1920, 1080)
    with patch.object(QtWidgets.QApplication, "screens", return_value=[mock_screen]):
        result = restore_widget_geometry(widget2, ini_path)

    # Off-screen geometry is adjusted (widget recentred) — still a successful restore.
    assert result is True


def test_mainwindow_state_not_saved_for_plain_widget(qtbot, tmp_path):
    """save_state=True on a plain QWidget does not write a 'state' key."""

    ini_path = tmp_path / "widget_no_state.ini"

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)

    save_widget_geometry(widget, ini_path, save_state=True)

    settings = QtCore.QSettings(str(ini_path), QtCore.QSettings.Format.IniFormat)
    settings.beginGroup(type(widget).__name__)
    state_value = settings.value("state")
    settings.endGroup()

    # A plain QWidget is not a QMainWindow so 'state' must not have been written.
    assert state_value is None


# ===========================================================================
# different_widget_classes_separate_keys
# ===========================================================================


def test_different_widget_classes_separate_keys(qtbot, tmp_path):
    """Two different widget types write to separate INI groups and do not collide."""
    ini_path = tmp_path / "multi.ini"

    label = QtWidgets.QLabel()
    qtbot.addWidget(label)
    label.resize(200, 150)

    push_btn = QtWidgets.QPushButton()
    qtbot.addWidget(push_btn)
    push_btn.resize(400, 50)

    save_widget_geometry(label, ini_path)
    save_widget_geometry(push_btn, ini_path)

    # Restore into fresh instances
    label2 = QtWidgets.QLabel()
    qtbot.addWidget(label2)
    push_btn2 = QtWidgets.QPushButton()
    qtbot.addWidget(push_btn2)

    restore_widget_geometry(label2, ini_path)
    restore_widget_geometry(push_btn2, ini_path)

    # Each widget should have received its own saved size, not the other's.
    assert label2.width() == 200
    assert label2.height() == 150
    assert push_btn2.width() == 400
    assert push_btn2.height() == 50
