"""Tests for DateFilterDialog widget."""

from unittest.mock import MagicMock

import pytest
from PySide6 import QtWidgets

from photoaident.core.date_range import DateRange
from photoaident.ui.widgets.date_filter_dialog import (
    DateFilterDialog,
    format_date_range,
)


@pytest.fixture
def mock_session_factory():
    """Return a session factory mock that returns no year range from the DB."""
    factory = MagicMock()
    session_ctx = MagicMock()
    # Make the context manager return a session
    factory.return_value.__enter__ = MagicMock(return_value=session_ctx)
    factory.return_value.__exit__ = MagicMock(return_value=False)
    # session.execute(...).one_or_none() returns None → dialog uses defaults
    session_ctx.execute.return_value.one_or_none.return_value = None
    return factory


def _select_year(combo: QtWidgets.QComboBox, year: int) -> None:
    """Helper: select a year in a year combobox by value."""
    idx = combo.findText(str(year))
    assert idx >= 0, f"Year {year} not found in year combo"
    combo.setCurrentIndex(idx)


class TestDateFilterDialogConstruction:
    def test_creates_without_error(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)

    def test_initial_state_all_not_set(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        assert dialog._start_year_combo.currentIndex() == 0
        assert dialog._end_year_combo.currentIndex() == 0
        assert dialog._start_month_combo.currentIndex() == 0
        assert dialog._end_month_combo.currentIndex() == 0

    def test_month_combos_disabled_when_year_not_set(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        assert not dialog._start_month_combo.isEnabled()
        assert not dialog._end_month_combo.isEnabled()

    def test_selected_range_none_initially(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        assert dialog.selected_range() is None

    def test_initial_range_populates_controls(self, qtbot, mock_session_factory):
        initial = DateRange(start_year=2020, start_month=3, end_year=2023, end_month=12)
        dialog = DateFilterDialog(mock_session_factory, initial_range=initial)
        qtbot.addWidget(dialog)
        assert dialog._start_year_combo.currentText() == "2020"
        assert dialog._start_month_combo.currentIndex() == 3  # March
        assert dialog._end_year_combo.currentText() == "2023"
        assert dialog._end_month_combo.currentIndex() == 12  # December

    def test_initial_range_enables_month_combos(self, qtbot, mock_session_factory):
        initial = DateRange(start_year=2020, end_year=2023)
        dialog = DateFilterDialog(mock_session_factory, initial_range=initial)
        qtbot.addWidget(dialog)
        assert dialog._start_month_combo.isEnabled()
        assert dialog._end_month_combo.isEnabled()

    def test_year_combo_contains_not_set_option(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        assert dialog._start_year_combo.itemText(0) == "(not set)"
        assert dialog._end_year_combo.itemText(0) == "(not set)"

    def test_year_combo_contains_years(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        assert dialog._start_year_combo.findText("1900") >= 0
        assert dialog._start_year_combo.findText("2020") >= 0

    def test_dialog_has_minimum_width(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        assert dialog.minimumWidth() >= 380


class TestMonthComboEnablement:
    def test_month_enabled_when_year_set(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        _select_year(dialog._start_year_combo, 2021)
        assert dialog._start_month_combo.isEnabled()

    def test_month_disabled_when_year_cleared(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        _select_year(dialog._start_year_combo, 2021)
        dialog._start_year_combo.setCurrentIndex(0)  # "(not set)"
        assert not dialog._start_month_combo.isEnabled()

    def test_month_resets_to_zero_when_year_cleared(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        _select_year(dialog._start_year_combo, 2021)
        dialog._start_month_combo.setCurrentIndex(5)  # May
        dialog._start_year_combo.setCurrentIndex(0)  # "(not set)"
        assert dialog._start_month_combo.currentIndex() == 0


class TestAcceptReject:
    def test_accept_with_no_values_sets_range_none(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        dialog._on_accept()
        assert dialog.selected_range() is None

    def test_accept_with_start_year_only(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        _select_year(dialog._start_year_combo, 2020)
        dialog._on_accept()
        result = dialog.selected_range()
        assert result is not None
        assert result.start_year == 2020
        assert result.start_month is None
        assert result.end_year is None

    def test_accept_with_full_range(self, qtbot, mock_session_factory):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        _select_year(dialog._start_year_combo, 2020)
        dialog._start_month_combo.setCurrentIndex(3)  # March
        _select_year(dialog._end_year_combo, 2023)
        dialog._end_month_combo.setCurrentIndex(12)  # December
        dialog._on_accept()
        result = dialog.selected_range()
        assert result is not None
        assert result.start_year == 2020
        assert result.start_month == 3
        assert result.end_year == 2023
        assert result.end_month == 12

    def test_validation_rejects_start_after_end(
        self, qtbot, mock_session_factory, monkeypatch
    ):
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        _select_year(dialog._start_year_combo, 2025)
        _select_year(dialog._end_year_combo, 2020)

        warning_shown = []

        def mock_warning(*args, **kwargs):
            warning_shown.append(True)

        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning", staticmethod(mock_warning)
        )
        dialog._on_accept()

        assert warning_shown, "QMessageBox.warning should have been shown"
        # Dialog should not have accepted (selected_range still None)
        assert dialog.selected_range() is None

    def test_cancel_leaves_range_unchanged(self, qtbot, mock_session_factory):
        initial = DateRange(start_year=2020)
        dialog = DateFilterDialog(mock_session_factory, initial_range=initial)
        qtbot.addWidget(dialog)
        dialog.reject()
        # selected_range was set from initial_range in __init__
        result = dialog.selected_range()
        assert result == initial

    def test_selecting_not_set_in_year_resets_selection(
        self, qtbot, mock_session_factory
    ):
        """Selecting '(not set)' year after a year was set clears the range."""
        dialog = DateFilterDialog(mock_session_factory)
        qtbot.addWidget(dialog)
        _select_year(dialog._start_year_combo, 2020)
        dialog._start_year_combo.setCurrentIndex(0)  # reset to "(not set)"
        dialog._on_accept()
        assert dialog.selected_range() is None


class TestFormatDateRange:
    def test_both_year_and_month(self):
        dr = DateRange(start_year=2020, start_month=3, end_year=2023, end_month=12)
        assert format_date_range(dr) == "Mar 2020 \u2013 Dec 2023"

    def test_year_only(self):
        dr = DateRange(start_year=2020, end_year=2023)
        assert format_date_range(dr) == "2020 \u2013 2023"

    def test_from_only(self):
        dr = DateRange(start_year=2020, start_month=3)
        assert format_date_range(dr) == "From Mar 2020"

    def test_until_only(self):
        dr = DateRange(end_year=2023, end_month=12)
        assert format_date_range(dr) == "Until Dec 2023"

    def test_start_year_no_month_until_month(self):
        dr = DateRange(start_year=2020, end_year=2023, end_month=6)
        assert format_date_range(dr) == "2020 \u2013 Jun 2023"
