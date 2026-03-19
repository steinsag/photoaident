"""Tests for FilepathDateWidget."""

from unittest.mock import MagicMock

from PySide6 import QtWidgets

from photoaident.core.filepath_date import PatternErrorCode
from photoaident.ui.widgets.filepath_date_widget import FilepathDateWidget

# ===========================================================================
# Initial state
# ===========================================================================


def test_defaults_disabled_empty_pattern(qtbot):
    """Widget defaults to disabled checkbox and empty pattern."""
    w = FilepathDateWidget()
    qtbot.addWidget(w)
    assert not w.is_enabled()
    assert w.pattern() == ""


def test_initial_enabled_and_pattern(qtbot):
    """Constructor arguments are reflected in widget state."""
    w = FilepathDateWidget(enabled=True, pattern="{YYYY}/{MM}/{DD}")
    qtbot.addWidget(w)
    assert w.is_enabled()
    assert w.pattern() == "{YYYY}/{MM}/{DD}"


def test_initial_disabled_pattern_field_disabled(qtbot):
    """Pattern field is disabled when the checkbox starts unchecked."""
    w = FilepathDateWidget(enabled=False)
    qtbot.addWidget(w)
    assert not w._pattern_edit.isEnabled()


def test_initial_enabled_pattern_field_enabled(qtbot):
    """Pattern field is enabled when the checkbox starts checked."""
    w = FilepathDateWidget(enabled=True)
    qtbot.addWidget(w)
    assert w._pattern_edit.isEnabled()


# ===========================================================================
# is_enabled / checkbox toggle
# ===========================================================================


def test_is_enabled_reflects_checkbox(qtbot):
    """is_enabled() tracks the checkbox state."""
    w = FilepathDateWidget(enabled=False)
    qtbot.addWidget(w)

    w._checkbox.setChecked(True)
    assert w.is_enabled()

    w._checkbox.setChecked(False)
    assert not w.is_enabled()


def test_checkbox_toggle_enables_pattern_field(qtbot):
    """Toggling the checkbox enables/disables the pattern field."""
    w = FilepathDateWidget(enabled=False)
    qtbot.addWidget(w)

    w._checkbox.setChecked(True)
    assert w._pattern_edit.isEnabled()

    w._checkbox.setChecked(False)
    assert not w._pattern_edit.isEnabled()


# ===========================================================================
# pattern()
# ===========================================================================


def test_pattern_returns_stripped_text(qtbot):
    """pattern() strips leading/trailing whitespace."""
    w = FilepathDateWidget(pattern="  {YYYY}-{MM}-{DD}  ")
    qtbot.addWidget(w)
    assert w.pattern() == "{YYYY}-{MM}-{DD}"


def test_pattern_returns_empty_when_blank(qtbot):
    """pattern() returns an empty string when the field is blank."""
    w = FilepathDateWidget(pattern="   ")
    qtbot.addWidget(w)
    assert w.pattern() == ""


# ===========================================================================
# validate()
# ===========================================================================


def test_validate_returns_true_when_unchecked(qtbot):
    """validate() is True regardless of pattern when unchecked."""
    w = FilepathDateWidget(enabled=False, pattern="")
    qtbot.addWidget(w)
    assert w.validate() is True


def test_validate_returns_true_when_unchecked_invalid_pattern(qtbot):
    """validate() skips pattern check when unchecked, even with bad pattern."""
    w = FilepathDateWidget(enabled=False, pattern="no-placeholders")
    qtbot.addWidget(w)
    assert w.validate() is True


def test_validate_returns_true_when_checked_and_valid(qtbot):
    """validate() returns True when checked and pattern is valid."""
    w = FilepathDateWidget(enabled=True, pattern="{YYYY}-{MM}-{DD}")
    qtbot.addWidget(w)
    assert w.validate() is True


def test_validate_returns_false_when_checked_and_empty(qtbot, monkeypatch):
    """validate() returns False and shows warning when pattern is empty."""
    w = FilepathDateWidget(enabled=True, pattern="")
    qtbot.addWidget(w)

    mock_warning = MagicMock()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", mock_warning)

    result = w.validate()

    assert result is False
    mock_warning.assert_called_once()


def test_validate_returns_false_when_pattern_missing_year(qtbot, monkeypatch):
    """validate() returns False when {YYYY} is absent."""
    w = FilepathDateWidget(enabled=True, pattern="{MM}-{DD}")
    qtbot.addWidget(w)

    mock_warning = MagicMock()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", mock_warning)

    assert w.validate() is False
    call_args = mock_warning.call_args[0]
    assert w.tr("Pattern must contain exactly one {YYYY} placeholder.") in call_args


def test_validate_does_not_show_warning_when_valid(qtbot, monkeypatch):
    """validate() does not call QMessageBox.warning on a valid pattern."""
    w = FilepathDateWidget(enabled=True, pattern="{YYYY}-{MM}-{DD}")
    qtbot.addWidget(w)

    mock_warning = MagicMock()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", mock_warning)

    w.validate()

    mock_warning.assert_not_called()


# ===========================================================================
# _translate_pattern_error — covers every PatternErrorCode
# ===========================================================================


def test_translate_pattern_error_empty(qtbot):
    w = FilepathDateWidget()
    qtbot.addWidget(w)
    assert w._translate_pattern_error(PatternErrorCode.EMPTY) == w.tr(
        "Pattern must not be empty."
    )


def test_translate_pattern_error_year_missing(qtbot):
    w = FilepathDateWidget()
    qtbot.addWidget(w)
    assert w._translate_pattern_error(PatternErrorCode.YEAR_MISSING) == w.tr(
        "Pattern must contain exactly one {YYYY} placeholder."
    )


def test_translate_pattern_error_covers_all_codes(qtbot):
    """_translate_pattern_error returns a non-empty string for every error code."""
    w = FilepathDateWidget()
    qtbot.addWidget(w)
    for code in PatternErrorCode:
        msg = w._translate_pattern_error(code)
        assert isinstance(msg, str) and msg
