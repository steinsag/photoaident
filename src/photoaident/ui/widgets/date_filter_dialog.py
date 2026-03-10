"""Dialog for selecting a date range filter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from PySide6 import QtWidgets
from sqlalchemy import func, select

from photoaident.core.date_range import DateRange
from photoaident.db.database import ImageMetadata

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

_MONTH_NAMES = [
    "",  # index 0 = "not set"
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_MONTH_ABBR = [
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

_YEAR_NOT_SET = 0
_YEAR_MIN = 1900
_YEAR_MAX = 2100


class DateFilterDialog(QtWidgets.QDialog):
    """Modal dialog for selecting a year/month date range.

    Queries the database for actual min/max taken_at to constrain the spinbox
    range. Supports open-ended ranges (start or end may be unset).
    """

    def __init__(
        self,
        session_factory: "sessionmaker",
        initial_range: Optional[DateRange] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Select Time Range"))
        self._selected_range: Optional[DateRange] = initial_range

        _, max_year = self._query_year_range(session_factory)
        self._setup_ui(initial_range, max_year)

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
        return _YEAR_MIN, _YEAR_MAX

    def _setup_ui(
        self,
        initial_range: Optional[DateRange],
        max_year: int,
    ) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        instruction = QtWidgets.QLabel(
            self.tr("Select the start and end of the time range to filter by.")
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        form_layout = QtWidgets.QFormLayout()

        self._start_year_spin, self._start_month_combo = self._make_year_month_row(
            max_year
        )
        self._end_year_spin, self._end_month_combo = self._make_year_month_row(max_year)

        self._start_year_spin.valueChanged.connect(self._on_start_year_changed)
        self._end_year_spin.valueChanged.connect(self._on_end_year_changed)

        from_row_widget = self._make_year_month_widget(
            self._start_year_spin, self._start_month_combo
        )
        to_row_widget = self._make_year_month_widget(
            self._end_year_spin, self._end_month_combo
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
                self._start_year_spin, self._start_month_combo
            )
            self._update_month_enabled_state(self._end_year_spin, self._end_month_combo)

    def _make_year_month_row(
        self, max_year: int
    ) -> tuple[QtWidgets.QSpinBox, QtWidgets.QComboBox]:
        """Create a year spinbox and month combobox for a row."""
        year_spin = QtWidgets.QSpinBox()
        year_spin.setRange(_YEAR_NOT_SET, max(max_year, _YEAR_MAX))
        year_spin.setMinimum(_YEAR_NOT_SET)
        year_spin.setSpecialValueText(self.tr("(not set)"))

        month_combo = QtWidgets.QComboBox()
        for name in _MONTH_NAMES:
            month_combo.addItem(self.tr(name) if name else "")

        return year_spin, month_combo

    def _make_year_month_widget(
        self,
        year_spin: QtWidgets.QSpinBox,
        month_combo: QtWidgets.QComboBox,
    ) -> QtWidgets.QWidget:
        """Wrap a year spinbox and month combo into a horizontal container."""
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(year_spin)
        row.addWidget(month_combo)
        return container

    def _apply_initial_range(self, date_range: DateRange) -> None:
        """Populate controls from an existing DateRange."""
        if date_range.start_year is not None:
            self._start_year_spin.setValue(date_range.start_year)
        if date_range.start_month is not None:
            self._start_month_combo.setCurrentIndex(date_range.start_month)
        else:
            self._start_month_combo.setCurrentIndex(0)

        if date_range.end_year is not None:
            self._end_year_spin.setValue(date_range.end_year)
        if date_range.end_month is not None:
            self._end_month_combo.setCurrentIndex(date_range.end_month)
        else:
            self._end_month_combo.setCurrentIndex(0)

        self._update_month_enabled_state(self._start_year_spin, self._start_month_combo)
        self._update_month_enabled_state(self._end_year_spin, self._end_month_combo)

    def _on_start_year_changed(self, _value: int) -> None:
        self._update_month_enabled_state(self._start_year_spin, self._start_month_combo)

    def _on_end_year_changed(self, _value: int) -> None:
        self._update_month_enabled_state(self._end_year_spin, self._end_month_combo)

    def _update_month_enabled_state(
        self,
        year_spin: QtWidgets.QSpinBox,
        month_combo: QtWidgets.QComboBox,
    ) -> None:
        """Disable month combo when year is not set; reset to empty when disabled."""
        enabled = year_spin.value() != _YEAR_NOT_SET
        month_combo.setEnabled(enabled)
        if not enabled:
            month_combo.setCurrentIndex(0)

    def _on_accept(self) -> None:
        start_year = self._start_year_spin.value() or None
        start_month_idx = self._start_month_combo.currentIndex()
        start_month = start_month_idx if start_month_idx > 0 and start_year else None

        end_year = self._end_year_spin.value() or None
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

    def selected_range(self) -> Optional[DateRange]:
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

    if date_range.start_year is not None:
        if date_range.start_month is not None:
            start_parts.append(_MONTH_ABBR[date_range.start_month])
        start_parts.append(str(date_range.start_year))

    if date_range.end_year is not None:
        if date_range.end_month is not None:
            end_parts.append(_MONTH_ABBR[date_range.end_month])
        end_parts.append(str(date_range.end_year))

    start_str = " ".join(start_parts)
    end_str = " ".join(end_parts)

    if start_str and end_str:
        return f"{start_str} \u2013 {end_str}"
    if start_str:
        return f"From {start_str}"
    if end_str:
        return f"Until {end_str}"
    return ""
