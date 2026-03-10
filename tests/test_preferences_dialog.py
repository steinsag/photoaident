from PySide6 import QtCore, QtWidgets

from photoaident.app import MainWindow
from photoaident.db.migrate import apply_migrations
from photoaident.settings import Settings
from photoaident.ui.preferences_dialog import PreferencesDialog


def test_preferences_dialog_initial_path(qtbot):
    dialog = PreferencesDialog("/initial/path", 10, 20)
    qtbot.add_widget(dialog)
    assert dialog.path_edit.text() == "/initial/path"
    assert dialog.image_count == 10
    assert dialog.face_count == 20


def test_preferences_dialog_accept(qtbot):
    dialog = PreferencesDialog("/initial/path", 10, 20)
    qtbot.add_widget(dialog)

    dialog.path_edit.setText("/new/path")

    # Click OK
    ok_button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert ok_button is not None
    qtbot.mouseClick(ok_button, QtCore.Qt.MouseButton.LeftButton)

    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert dialog.get_collection_path() == "/new/path"


def test_preferences_dialog_reject(qtbot):
    dialog = PreferencesDialog("/initial/path", 10, 20)
    qtbot.add_widget(dialog)

    dialog.path_edit.setText("/new/path")

    cancel_button = dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    assert cancel_button is not None
    qtbot.mouseClick(cancel_button, QtCore.Qt.MouseButton.LeftButton)

    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert dialog.get_collection_path() == "/new/path"


def test_preferences_save_settings(qtbot, tmp_app_paths, monkeypatch):
    """Test that settings are saved when the dialog is accepted in MainWindow."""
    # Apply migrations to the test DB
    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")

    # Create a MainWindow
    window = MainWindow(tmp_app_paths, check_gpu=False, enable_onboarding=False)
    qtbot.add_widget(window)

    # Mock PreferencesDialog.exec to simulate OK
    def mock_exec(self):
        self.path_edit.setText("/mock/saved/path")
        return QtWidgets.QDialog.DialogCode.Accepted

    # Mock ProgressDialog.exec to avoid modal blocking in tests
    def mock_progress_exec(self):
        # We don't actually want to run the thread in this test,
        # but let's just make it return immediately.
        return QtWidgets.QDialog.DialogCode.Accepted

    # Mock QMessageBox.question to simulate Yes
    def mock_question(*args, **kwargs):
        return QtWidgets.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(PreferencesDialog, "exec", mock_exec)
    from photoaident.ui.widgets.progress_dialog import ProgressDialog

    monkeypatch.setattr(ProgressDialog, "exec", mock_progress_exec)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", mock_question)

    # Show preferences (this calls dialog.exec() and then saves if Accepted)
    window._show_preferences()

    # Verify settings in memory
    assert window._settings.collection_path == "/mock/saved/path"

    # Verify settings on disk
    loaded_settings = Settings.load(tmp_app_paths.config_file)
    assert loaded_settings.collection_path == "/mock/saved/path"


def test_browse_path_updates_edit(qtbot, monkeypatch):
    """_browse_path updates path_edit when a directory is chosen."""
    dialog = PreferencesDialog("/initial/path", 0, 0)
    qtbot.add_widget(dialog)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *_: "/chosen/path",
    )

    dialog._browse_path()

    assert dialog.path_edit.text() == "/chosen/path"


def test_browse_path_cancelled_keeps_edit(qtbot, monkeypatch):
    """_browse_path leaves path_edit unchanged when the dialog is cancelled."""
    dialog = PreferencesDialog("/initial/path", 0, 0)
    qtbot.add_widget(dialog)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *_: "",
    )

    dialog._browse_path()

    assert dialog.path_edit.text() == "/initial/path"


def test_preferences_cancel_not_saved(qtbot, tmp_app_paths, monkeypatch):
    """Test that settings are NOT saved when the dialog is cancelled."""
    # Apply migrations to the test DB
    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")

    initial_path = "/initial/path"
    tmp_app_paths.config_file.parent.mkdir(parents=True, exist_ok=True)
    Settings(collection_path=initial_path).save(tmp_app_paths.config_file)

    window = MainWindow(tmp_app_paths, check_gpu=False, enable_onboarding=False)
    qtbot.add_widget(window)

    # Mock PreferencesDialog.exec to simulate Cancel
    def mock_exec(self):
        self.path_edit.setText("/mock/cancelled/path")
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(PreferencesDialog, "exec", mock_exec)

    # Show preferences
    window._show_preferences()

    # Verify settings in memory NOT updated
    assert window._settings.collection_path == initial_path

    # Verify settings on disk NOT updated
    loaded_settings = Settings.load(tmp_app_paths.config_file)
    assert loaded_settings.collection_path == initial_path
