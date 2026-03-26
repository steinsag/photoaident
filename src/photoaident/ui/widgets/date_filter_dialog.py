"""Dialog for selecting a date range filter."""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

from PySide6 import QtWidgets
from PySide6.QtCore import QCoreApplication, QLocale
from sqlalchemy import func, select

from photoaident.core.date_range import DateRange
from photoaident.db.database import ImageMetadata

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

_YEAR_MIN = 1900
_DIALOG_MIN_WIDTH = 380


class DateFilterDialog(QtWidgets.QDialog):
    """Modal dialog for selecting a year/month date range.

    Queries the database for actual min/max taken_at to populate the year
    combobox. Supports open-ended ranges (start or end may be unset). Selecting
    "(not set)" in the year combobox clears the year and disables the month.
    """

    def __init__(
        self,
        session_factory: "sessionmaker",
        initial_range: DateRange | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Select Time Range"))
        self.setMinimumWidth(_DIALOG_MIN_WIDTH)
        self._selected_range: DateRange | None = initial_range

        min_year, max_year = self._query_year_range(session_factory)
        self._setup_ui(initial_range, min_year, max_year)

    def _query_year_range(self, session_factory: "sessionmaker") -> tuple[int, int]:
        """Query the database for the min/max year of taken_at."""
        try:
            with session_factory() as session:
                row = session.execute(
                    select(
                        func.min(func.strftime("%Y", ImageMetadata.taken_at)),
                        func.max(func.strftime("%Y", ImageMetadata.taken_at)),
                    )
                ).one_or_none()
                if row and row[0] and row[1]:
                    return int(row[0]), int(row[1])
        except Exception:
            logger.exception("Failed to query year range from database")
        return _YEAR_MIN, datetime.date.today().year

    def _setup_ui(
        self,
        initial_range: DateRange | None,
        min_year: int,
        max_year: int,
    ) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        instruction = QtWidgets.QLabel(
            self.tr("Select the start and end of the time range to filter by.")
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        form_layout = QtWidgets.QFormLayout()

        self._start_year_combo, self._start_month_combo = self._make_year_month_row(
            min_year, max_year
        )
        self._end_year_combo, self._end_month_combo = self._make_year_month_row(
            min_year, max_year
        )

        self._start_year_combo.currentIndexChanged.connect(self._on_start_year_changed)
        self._end_year_combo.currentIndexChanged.connect(self._on_end_year_changed)

        from_row_widget = self._make_year_month_widget(
            self._start_year_combo, self._start_month_combo
        )
        to_row_widget = self._make_year_month_widget(
            self._end_year_combo, self._end_month_combo
        )

        form_layout.addRow(self.tr("From:"), from_row_widget)
        form_layout.addRow(self.tr("To:"), to_row_widget)
        layout.addLayout(form_layout)

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

        if initial_range is not None:
            self._apply_initial_range(initial_range)
        else:
            self._update_month_enabled_state(
                self._start_year_combo, self._start_month_combo
            )
            self._update_month_enabled_state(
                self._end_year_combo, self._end_month_combo
            )

    def _make_year_month_row(
        self, min_year: int, max_year: int
    ) -> tuple[QtWidgets.QComboBox, QtWidgets.QComboBox]:
        """Create a year combobox and month combobox for a row.

        Index 0 of the year combobox is "(not set)"; subsequent items are years
        from min_year to max_year. Selecting "(not set)" disables the month combo.
        """
        year_combo = QtWidgets.QComboBox()
        year_combo.addItem(self.tr("(not set)"))  # index 0
        for year in range(min_year, max_year + 1):
            year_combo.addItem(str(year))

        locale = QLocale()
        month_combo = QtWidgets.QComboBox()
        month_combo.addItem("")  # index 0 = "not set"
        for month in range(1, 13):
            month_combo.addItem(
                locale.standaloneMonthName(month, QLocale.FormatType.LongFormat)
            )

        return year_combo, month_combo

    def _make_year_month_widget(
        self,
        year_combo: QtWidgets.QComboBox,
        month_combo: QtWidgets.QComboBox,
    ) -> QtWidgets.QWidget:
        """Wrap a year combobox and month combo into a horizontal container."""
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(year_combo)
        row.addWidget(month_combo)
        return container

    def _apply_initial_range(self, date_range: DateRange) -> None:
        """Populate controls from an existing DateRange."""
        self._set_year_combo(self._start_year_combo, date_range.start_year)
        self._start_month_combo.setCurrentIndex(date_range.start_month or 0)

        self._set_year_combo(self._end_year_combo, date_range.end_year)
        self._end_month_combo.setCurrentIndex(date_range.end_month or 0)

        self._update_month_enabled_state(
            self._start_year_combo, self._start_month_combo
        )
        self._update_month_enabled_state(self._end_year_combo, self._end_month_combo)

    @staticmethod
    def _set_year_combo(combo: QtWidgets.QComboBox, year: int | None) -> None:
        """Select a year in the combo, or select '(not set)' if year is None."""
        if year is None:
            combo.setCurrentIndex(0)
            return
        idx = combo.findText(str(year))
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    @staticmethod
    def _get_year_from_combo(combo: QtWidgets.QComboBox) -> int | None:
        """Return the selected year, or None if '(not set)' is selected."""
        if combo.currentIndex() == 0:
            return None
        return int(combo.currentText())

    def _on_start_year_changed(self, _index: int) -> None:
        self._update_month_enabled_state(
            self._start_year_combo, self._start_month_combo
        )

    def _on_end_year_changed(self, _index: int) -> None:
        self._update_month_enabled_state(self._end_year_combo, self._end_month_combo)

    def _update_month_enabled_state(
        self,
        year_combo: QtWidgets.QComboBox,
        month_combo: QtWidgets.QComboBox,
    ) -> None:
        """Disable month combo when year is not set; reset to empty when disabled."""
        enabled = year_combo.currentIndex() != 0
        month_combo.setEnabled(enabled)
        if not enabled:
            month_combo.setCurrentIndex(0)

    def _on_accept(self) -> None:
        start_year = self._get_year_from_combo(self._start_year_combo)
        start_month_idx = self._start_month_combo.currentIndex()
        start_month = start_month_idx if start_month_idx > 0 and start_year else None

        end_year = self._get_year_from_combo(self._end_year_combo)
        end_month_idx = self._end_month_combo.currentIndex()
        end_month = end_month_idx if end_month_idx > 0 and end_year else None

        if start_year is not None or end_year is not None:
            try:
                candidate = DateRange(
                    start_year=start_year,
                    start_month=start_month,
                    end_year=end_year,
                    end_month=end_month,
                )
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Invalid Date Range"),
                    self.tr("The selected date range is invalid: {error}").format(
                        error=str(exc)
                    ),
                )
                return
            self._selected_range = candidate
        else:
            self._selected_range = None
        self.accept()

    def selected_range(self) -> DateRange | None:
        """Return the selected DateRange, or None if no range was set."""
        return self._selected_range


def format_date_range(date_range: DateRange) -> str:
    """Format a DateRange for display in a button label.

    Examples:
        "Mar 2020 – Dec 2023"
        "2020 – Dec 2023"
        "From Mar 2020"
        "Until Dec 2023"
        "2020 – 2023"
    """
    start_parts: list[str] = []
    end_parts: list[str] = []

    locale = QLocale()

    if date_range.start_year is not None:
        if date_range.start_month is not None:
            start_parts.append(
                locale.standaloneMonthName(
                    date_range.start_month, QLocale.FormatType.ShortFormat
                )
            )
        start_parts.append(str(date_range.start_year))

    if date_range.end_year is not None:
        if date_range.end_month is not None:
            end_parts.append(
                locale.standaloneMonthName(
                    date_range.end_month, QLocale.FormatType.ShortFormat
                )
            )
        end_parts.append(str(date_range.end_year))

    start_str = " ".join(start_parts)
    end_str = " ".join(end_parts)

    if start_str and end_str:
        return f"{start_str} \u2013 {end_str}"
    if start_str:
        return QCoreApplication.translate("DateFilterDialog", "From {start}").format(
            start=start_str
        )
    if end_str:
        return QCoreApplication.translate("DateFilterDialog", "Until {end}").format(
            end=end_str
        )
    return ""
