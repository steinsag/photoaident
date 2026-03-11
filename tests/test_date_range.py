"""Unit tests for photoaident.core.date_range.DateRange."""

import pytest
from datetime import datetime

from photoaident.core.date_range import DateRange


class TestToStartDatetime:
    def test_no_start_year_returns_none(self):
        dr = DateRange()
        assert dr.to_start_datetime() is None

    def test_year_only_gives_jan_first(self):
        dr = DateRange(start_year=2020)
        assert dr.to_start_datetime() == datetime(2020, 1, 1, 0, 0, 0)

    def test_year_and_month_gives_first_of_month(self):
        dr = DateRange(start_year=2020, start_month=3)
        assert dr.to_start_datetime() == datetime(2020, 3, 1, 0, 0, 0)

    def test_december_start(self):
        dr = DateRange(start_year=2023, start_month=12)
        assert dr.to_start_datetime() == datetime(2023, 12, 1, 0, 0, 0)


class TestToEndDatetime:
    def test_no_end_year_returns_none(self):
        dr = DateRange()
        assert dr.to_end_datetime() is None

    def test_year_only_gives_dec_31_last_microsecond(self):
        dr = DateRange(end_year=2023)
        assert dr.to_end_datetime() == datetime(2023, 12, 31, 23, 59, 59, 999999)

    def test_year_and_month_gives_last_day_of_month(self):
        dr = DateRange(end_year=2023, end_month=3)
        assert dr.to_end_datetime() == datetime(2023, 3, 31, 23, 59, 59, 999999)

    def test_february_non_leap_year(self):
        dr = DateRange(end_year=2023, end_month=2)
        assert dr.to_end_datetime() == datetime(2023, 2, 28, 23, 59, 59, 999999)

    def test_february_leap_year(self):
        dr = DateRange(end_year=2024, end_month=2)
        assert dr.to_end_datetime() == datetime(2024, 2, 29, 23, 59, 59, 999999)

    def test_april_thirty_days(self):
        dr = DateRange(end_year=2022, end_month=4)
        assert dr.to_end_datetime() == datetime(2022, 4, 30, 23, 59, 59, 999999)


class TestValidation:
    def test_invalid_start_month_zero(self):
        with pytest.raises(ValueError, match="start_month"):
            DateRange(start_year=2020, start_month=0)

    def test_invalid_start_month_thirteen(self):
        with pytest.raises(ValueError, match="start_month"):
            DateRange(start_year=2020, start_month=13)

    def test_invalid_end_month_zero(self):
        with pytest.raises(ValueError, match="end_month"):
            DateRange(end_year=2020, end_month=0)

    def test_invalid_end_month_thirteen(self):
        with pytest.raises(ValueError, match="end_month"):
            DateRange(end_year=2020, end_month=13)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="start date"):
            DateRange(start_year=2025, end_year=2020)

    def test_start_after_end_same_year_month_raises(self):
        with pytest.raises(ValueError, match="start date"):
            DateRange(start_year=2020, start_month=6, end_year=2020, end_month=3)

    def test_same_month_is_valid(self):
        # start == end is allowed (single month)
        dr = DateRange(start_year=2020, start_month=6, end_year=2020, end_month=6)
        assert dr.to_start_datetime() == datetime(2020, 6, 1, 0, 0, 0)
        assert dr.to_end_datetime() == datetime(2020, 6, 30, 23, 59, 59, 999999)

    def test_open_ended_no_start(self):
        dr = DateRange(end_year=2023)
        assert dr.to_start_datetime() is None
        assert dr.to_end_datetime() is not None

    def test_open_ended_no_end(self):
        dr = DateRange(start_year=2020)
        assert dr.to_start_datetime() is not None
        assert dr.to_end_datetime() is None

    def test_fully_open_is_valid(self):
        dr = DateRange()
        assert dr.to_start_datetime() is None
        assert dr.to_end_datetime() is None

    def test_month_without_year_still_validates(self):
        # start_month set without start_year is allowed (year is open-ended)
        dr = DateRange(start_month=3)
        assert dr.to_start_datetime() is None  # year is None so returns None

    def test_frozen(self):
        dr = DateRange(start_year=2020)
        with pytest.raises(Exception):
            dr.start_year = 2021  # type: ignore[misc]
