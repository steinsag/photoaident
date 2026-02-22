from PySide6 import QtCore

from photoaident.ui.widgets.progress_dialog import ProgressDialog


def test_progress_dialog_initial_state(qtbot):
    dialog = ProgressDialog("Test Title", "Initial Message")
    qtbot.add_widget(dialog)

    assert dialog.windowTitle() == "Test Title"
    assert dialog.label.text() == "Initial Message"
    assert dialog.isModal()
    assert dialog.progress_bar.maximum() == 0  # Indeterminate range
    assert not (dialog.windowFlags() & QtCore.Qt.WindowType.WindowCloseButtonHint)


def test_progress_dialog_update_status(qtbot):
    dialog = ProgressDialog("Title", "Old Status")
    qtbot.add_widget(dialog)

    dialog.update_status("New Status")
    assert dialog.label.text() == "New Status"


def test_progress_dialog_update_progress(qtbot):
    dialog = ProgressDialog("Title", "Message")
    qtbot.add_widget(dialog)

    dialog.update_progress(5, 100)
    assert dialog.progress_bar.maximum() == 100
    assert dialog.progress_bar.value() == 5

    # Check that it updates maximum correctly if it changes
    dialog.update_progress(50, 200)
    assert dialog.progress_bar.maximum() == 200
    assert dialog.progress_bar.value() == 50
