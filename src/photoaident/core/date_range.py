"""Date range value object for filtering images by when they were taken."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DateRange:
    """A date range for filtering images by taken_at timestamp.

    All fields are optional, supporting open-ended ranges. Months use 1–12.
    """

    start_year: int | None = None
    start_month: int | None = None  # 1-12, optional
    end_year: int | None = None
    end_month: int | None = None  # 1-12, optional

    def __post_init__(self) -> None:
        for field_name, value in [
            ("start_month", self.start_month),
            ("end_month", self.end_month),
        ]:
            if value is not None and not (1 <= value <= 12):
                raise ValueError(f"{field_name} must be between 1 and 12, got {value}")

        start_dt = self.to_start_datetime()
        end_dt = self.to_end_datetime()
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError("start date must not be after end date")

    def to_start_datetime(self) -> datetime | None:
        """Return the first moment of the start date, or None if no start year set."""
        if self.start_year is None:
            return None
        month = self.start_month if self.start_month is not None else 1
        return datetime(self.start_year, month, 1, 0, 0, 0)

    def to_end_datetime(self) -> datetime | None:
        """Return the last moment of the end date, or None if no end year is set."""
        if self.end_year is None:
            return None
        month = self.end_month if self.end_month is not None else 12
        last_day = calendar.monthrange(self.end_year, month)[1]
        return datetime(self.end_year, month, last_day, 23, 59, 59, 999999)
