from unittest.mock import MagicMock

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

    monkeypatch.setattr(PreferencesDialog, "exec", mock_exec)
    from photoaident.ui.widgets.progress_dialog import ProgressDialog

    monkeypatch.setattr(
        ProgressDialog,
        "exec",
        lambda *_: QtWidgets.QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_: QtWidgets.QMessageBox.StandardButton.Yes,
    )

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


# ===========================================================================
# _validate_pattern_message
# ===========================================================================


def test_validate_pattern_message_empty_returns_error(qtbot):
    """Empty string returns the 'must not be empty' error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("")
    assert result == dialog.tr("Pattern must not be empty.")


def test_validate_pattern_message_missing_yyyy_returns_error(qtbot):
    """Pattern without {YYYY} returns the missing-year error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{MM}-{DD}")
    assert result == dialog.tr("Pattern must contain exactly one {YYYY} placeholder.")


def test_validate_pattern_message_duplicate_yyyy_returns_error(qtbot):
    """Pattern with two {YYYY} returns the duplicate-YYYY error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}{YYYY}{M}{D}")
    assert result == dialog.tr("Pattern must not contain {YYYY} more than once.")


def test_validate_pattern_message_duplicate_mm_returns_error(qtbot):
    """Pattern with two {MM} returns the duplicate-MM error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}{MM}{MM}{D}")
    assert result == dialog.tr("Pattern must not contain {MM} more than once.")


def test_validate_pattern_message_duplicate_m_returns_error(qtbot):
    """Pattern with two {M} returns the duplicate-M error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}{M}{M}{DD}")
    assert result == dialog.tr("Pattern must not contain {M} more than once.")


def test_validate_pattern_message_duplicate_dd_returns_error(qtbot):
    """Pattern with two {DD} returns the duplicate-DD error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}{M}{DD}{DD}")
    assert result == dialog.tr("Pattern must not contain {DD} more than once.")


def test_validate_pattern_message_duplicate_d_returns_error(qtbot):
    """Pattern with two {D} returns the duplicate-D error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}{MM}{D}{D}")
    assert result == dialog.tr("Pattern must not contain {D} more than once.")


def test_validate_pattern_message_both_mm_and_m_returns_error(qtbot):
    """Pattern with both {MM} and {M} returns the conflicting-month error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}-{MM}-{M}-{DD}")
    assert result == dialog.tr("Pattern must not contain both {MM} and {M}.")


def test_validate_pattern_message_no_month_placeholder_returns_error(qtbot):
    """Pattern with no month placeholder returns the missing-month error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}-{DD}")
    assert result == dialog.tr(
        "Pattern must contain a month placeholder ({MM} or {M})."
    )


def test_validate_pattern_message_both_dd_and_d_returns_error(qtbot):
    """Pattern with both {DD} and {D} returns the conflicting-day error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}-{MM}-{DD}-{D}")
    assert result == dialog.tr("Pattern must not contain both {DD} and {D}.")


def test_validate_pattern_message_no_day_placeholder_returns_error(qtbot):
    """Pattern with no day placeholder returns the missing-day error message."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}-{MM}")
    assert result == dialog.tr("Pattern must contain a day placeholder ({DD} or {D}).")


def test_validate_pattern_message_valid_full_pattern_returns_none(qtbot):
    """Valid pattern with all required placeholders returns None."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{YYYY}-{MM}-{DD}")
    assert result is None


def test_validate_pattern_message_valid_single_digit_variants_returns_none(qtbot):
    """Valid pattern using {M} and {D} single-digit variants returns None."""
    dialog = PreferencesDialog("/path", 0, 0)
    qtbot.add_widget(dialog)
    result = dialog._validate_pattern_message("{DD}.{M}.{YYYY}")
    assert result is None


# ===========================================================================
# accept
# ===========================================================================


def test_accept_blocks_when_checkbox_checked_and_pattern_invalid(qtbot, monkeypatch):
    """accept() does not close the dialog when pattern is invalid."""
    dialog = PreferencesDialog(
        "/path", 0, 0, filepath_date_enabled=True, filepath_date_pattern=""
    )
    qtbot.add_widget(dialog)

    mock_warning = MagicMock()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", mock_warning)
    dialog.accept()

    mock_warning.assert_called_once()
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted


def test_accept_shows_warning_with_error_message_when_invalid(qtbot, monkeypatch):
    """accept() passes the validation error text to QMessageBox.warning."""
    dialog = PreferencesDialog(
        "/path", 0, 0, filepath_date_enabled=True, filepath_date_pattern="{MM}-{DD}"
    )
    qtbot.add_widget(dialog)

    mock_warning = MagicMock()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", mock_warning)
    dialog.accept()

    call_args = mock_warning.call_args[0]
    assert (
        dialog.tr("Pattern must contain exactly one {YYYY} placeholder.") in call_args
    )


def test_accept_succeeds_when_checkbox_checked_and_pattern_valid(qtbot):
    """accept() closes the dialog when the checkbox is checked and pattern is valid."""
    dialog = PreferencesDialog(
        "/path",
        0,
        0,
        filepath_date_enabled=True,
        filepath_date_pattern="{YYYY}-{MM}-{DD}",
    )
    qtbot.add_widget(dialog)

    dialog.accept()

    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_accept_succeeds_when_checkbox_unchecked_regardless_of_pattern(qtbot):
    """accept() closes the dialog when unchecked even if pattern is empty."""
    dialog = PreferencesDialog(
        "/path", 0, 0, filepath_date_enabled=False, filepath_date_pattern=""
    )
    qtbot.add_widget(dialog)

    dialog.accept()

    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_accept_unchecked_does_not_show_warning(qtbot, monkeypatch):
    """accept() never calls QMessageBox.warning when the checkbox is unchecked."""
    dialog = PreferencesDialog(
        "/path", 0, 0, filepath_date_enabled=False, filepath_date_pattern=""
    )
    qtbot.add_widget(dialog)

    mock_warning = MagicMock()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", mock_warning)
    dialog.accept()

    mock_warning.assert_not_called()
