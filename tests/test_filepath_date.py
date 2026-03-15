"""Unit tests for photoaident.core.filepath_date."""

import re
from datetime import date
from pathlib import Path

import pytest

from photoaident.core.filepath_date import (
    PatternErrorCode,
    PatternValidationError,
    compile_pattern,
    extract_date_from_path,
)

# ===========================================================================
# compile_pattern — valid patterns
# ===========================================================================


class TestCompilePatternValid:
    def test_yyyy_mm_dd_returns_pattern(self):
        """compile_pattern with {YYYY}-{MM}-{DD} returns a compiled re.Pattern."""
        result = compile_pattern("{YYYY}-{MM}-{DD}")
        assert isinstance(result, re.Pattern)

    def test_yyyy_mm_dd_has_named_groups(self):
        """compile_pattern produces groups year, month, day."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        match = pattern.search("2026-01-17")
        assert match is not None
        assert match.group("year") == "2026"
        assert match.group("month") == "01"
        assert match.group("day") == "17"

    def test_dd_dot_m_dot_yyyy_single_digit_month(self):
        """compile_pattern with {DD}.{M}.{YYYY} accepts 1-digit month."""
        pattern = compile_pattern("{DD}.{M}.{YYYY}")
        match = pattern.search("17.1.2026")
        assert match is not None
        assert match.group("year") == "2026"
        assert match.group("month") == "1"
        assert match.group("day") == "17"

    def test_dd_dot_m_dot_yyyy_two_digit_month(self):
        """compile_pattern with {DD}.{M}.{YYYY} also accepts 2-digit month."""
        pattern = compile_pattern("{DD}.{M}.{YYYY}")
        match = pattern.search("17.11.2026")
        assert match is not None
        assert match.group("month") == "11"

    def test_pxl_prefix_no_separators(self):
        """compile_pattern with PXL{YYYY}{MM}{DD} matches a compact filename."""
        pattern = compile_pattern("PXL{YYYY}{MM}{DD}")
        match = pattern.search("PXL20260117")
        assert match is not None
        assert match.group("year") == "2026"
        assert match.group("month") == "01"
        assert match.group("day") == "17"

    def test_yyyy_m_dd_variant(self):
        """compile_pattern with {YYYY}-{M}-{DD} returns a compiled re.Pattern."""
        pattern = compile_pattern("{YYYY}-{M}-{DD}")
        assert isinstance(pattern, re.Pattern)
        match = pattern.search("2026-3-07")
        assert match is not None
        assert match.group("month") == "3"
        assert match.group("day") == "07"

    def test_yyyy_mm_d_variant(self):
        """compile_pattern with {YYYY}.{MM}.{D} returns a compiled re.Pattern."""
        pattern = compile_pattern("{YYYY}.{MM}.{D}")
        assert isinstance(pattern, re.Pattern)
        match = pattern.search("2026.08.5")
        assert match is not None
        assert match.group("day") == "5"

    def test_yyyy_m_d_variant(self):
        """compile_pattern with {YYYY}/{M}/{D} returns a compiled re.Pattern."""
        pattern = compile_pattern("{YYYY}/{M}/{D}")
        assert isinstance(pattern, re.Pattern)
        match = pattern.search("2026/3/9")
        assert match is not None
        assert match.group("month") == "3"
        assert match.group("day") == "9"


# ===========================================================================
# compile_pattern — literal escaping
# ===========================================================================


class TestCompilePatternLiteralEscaping:
    def test_dot_is_literal_not_wildcard(self):
        """Dots in the pattern match literal dots, not any character."""
        pattern = compile_pattern("{YYYY}.{MM}.{DD}")
        assert pattern.search("2026X01X17") is None

    def test_dot_matches_literal_dot(self):
        """Dots in the pattern match a literal dot in the input."""
        pattern = compile_pattern("{YYYY}.{MM}.{DD}")
        assert pattern.search("2026.01.17") is not None

    def test_slash_is_literal(self):
        """Forward slash in the pattern matches a literal slash."""
        pattern = compile_pattern("{YYYY}/{MM}/{DD}")
        assert pattern.search("2026/01/17") is not None
        assert pattern.search("20260117") is None

    def test_special_regex_chars_escaped(self):
        """Pattern containing regex special characters are treated as literals."""
        pattern = compile_pattern("{YYYY}({MM}){DD}")
        assert pattern.search("2026(01)17") is not None
        assert pattern.search("20260117") is None


# ===========================================================================
# compile_pattern — invalid patterns (error cases)
# ===========================================================================


class TestCompilePatternInvalid:
    def test_empty_string_raises_pattern_validation_error(self):
        """compile_pattern('') raises PatternValidationError with EMPTY code."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("")
        assert exc_info.value.code is PatternErrorCode.EMPTY

    def test_missing_yyyy_raises_year_missing_code(self):
        """compile_pattern without {YYYY} raises YEAR_MISSING."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{MM}-{DD}")
        assert exc_info.value.code is PatternErrorCode.YEAR_MISSING

    def test_duplicate_yyyy_raises_year_duplicate_code(self):
        """compile_pattern with two {YYYY} raises YEAR_DUPLICATE."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}{YYYY}{M}{D}")
        assert exc_info.value.code is PatternErrorCode.YEAR_DUPLICATE

    def test_both_mm_and_m_raises_month_conflict_code(self):
        """compile_pattern with both {MM} and {M} raises MONTH_CONFLICT."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}-{MM}-{M}-{DD}")
        assert exc_info.value.code is PatternErrorCode.MONTH_CONFLICT

    def test_missing_month_placeholder_raises_month_missing_code(self):
        """compile_pattern without any month placeholder raises MONTH_MISSING."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}-{DD}")
        assert exc_info.value.code is PatternErrorCode.MONTH_MISSING

    def test_duplicate_mm_raises_month_mm_duplicate_code(self):
        """compile_pattern with two {MM} raises MONTH_MM_DUPLICATE."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}{MM}{MM}{D}")
        assert exc_info.value.code is PatternErrorCode.MONTH_MM_DUPLICATE

    def test_duplicate_m_raises_month_m_duplicate_code(self):
        """compile_pattern with two {M} raises MONTH_M_DUPLICATE."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}{M}{M}{DD}")
        assert exc_info.value.code is PatternErrorCode.MONTH_M_DUPLICATE

    def test_both_dd_and_d_raises_day_conflict_code(self):
        """compile_pattern with both {DD} and {D} raises DAY_CONFLICT."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}-{MM}-{DD}-{D}")
        assert exc_info.value.code is PatternErrorCode.DAY_CONFLICT

    def test_missing_day_placeholder_raises_day_missing_code(self):
        """compile_pattern without any day placeholder raises DAY_MISSING."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}-{MM}")
        assert exc_info.value.code is PatternErrorCode.DAY_MISSING

    def test_duplicate_dd_raises_day_dd_duplicate_code(self):
        """compile_pattern with two {DD} raises DAY_DD_DUPLICATE."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}{M}{DD}{DD}")
        assert exc_info.value.code is PatternErrorCode.DAY_DD_DUPLICATE

    def test_duplicate_d_raises_day_d_duplicate_code(self):
        """compile_pattern with two {D} raises DAY_D_DUPLICATE."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}{MM}{D}{D}")
        assert exc_info.value.code is PatternErrorCode.DAY_D_DUPLICATE

    def test_only_yyyy_raises_month_missing_code(self):
        """compile_pattern with only {YYYY} raises MONTH_MISSING."""
        with pytest.raises(PatternValidationError) as exc_info:
            compile_pattern("{YYYY}")
        assert exc_info.value.code is PatternErrorCode.MONTH_MISSING

    def test_pattern_validation_error_is_value_error(self):
        """PatternValidationError is a subclass of ValueError."""
        with pytest.raises(ValueError):
            compile_pattern("")


# ===========================================================================
# extract_date_from_path — happy path (docstring examples)
# ===========================================================================


class TestExtractDateFromPathHappyPath:
    def test_iso_format_in_filename(self):
        """ISO date in filename is extracted correctly."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(
            Path("/photos/vacation/2026-01-17.jpg"), pattern
        )
        assert result == date(2026, 1, 17)

    def test_european_format_single_digit_month(self):
        """European single-digit month date in filename is extracted correctly."""
        pattern = compile_pattern("{DD}.{M}.{YYYY}")
        result = extract_date_from_path(Path("/photos/17.1.2026_holiday.jpg"), pattern)
        assert result == date(2026, 1, 17)

    def test_pxl_compact_format_in_filename(self):
        """Compact Google Pixel filename date is extracted correctly."""
        pattern = compile_pattern("PXL{YYYY}{MM}{DD}")
        result = extract_date_from_path(
            Path("/sdcard/DCIM/Camera/PXL20260117_123456.jpg"), pattern
        )
        assert result == date(2026, 1, 17)

    def test_date_in_directory_part_of_path(self):
        """Date embedded in directory component of path is found."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/archive/2026-01-17/img001.jpg"), pattern)
        assert result == date(2026, 1, 17)

    def test_date_in_filename_not_in_directory(self):
        """Date in filename is found when directory contains no date."""
        pattern = compile_pattern("{YYYY}{MM}{DD}")
        result = extract_date_from_path(
            Path("/photos/misc/20260117_sunset.jpg"), pattern
        )
        assert result == date(2026, 1, 17)

    def test_december_last_day(self):
        """Dec 31 is extracted correctly."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2023-12-31.jpg"), pattern)
        assert result == date(2023, 12, 31)

    def test_january_first(self):
        """Jan 1 is extracted correctly."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2023-01-01.jpg"), pattern)
        assert result == date(2023, 1, 1)

    def test_two_digit_year_not_confused_with_four(self):
        """Pattern requires exactly 4-digit year and does not match 2-digit year."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/26-01-17.jpg"), pattern)
        assert result is None

    def test_single_digit_day_with_d_placeholder(self):
        """Single-digit day matched by {D} placeholder."""
        pattern = compile_pattern("{YYYY}-{MM}-{D}")
        result = extract_date_from_path(Path("/pics/2026-03-7.jpg"), pattern)
        assert result == date(2026, 3, 7)


# ===========================================================================
# extract_date_from_path — invalid dates return None
# ===========================================================================


class TestExtractDateFromPathInvalidDates:
    def test_february_30_returns_none(self):
        """Feb 30 does not exist and returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2026-02-30.jpg"), pattern)
        assert result is None

    def test_november_31_returns_none(self):
        """Nov 31 does not exist and returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2026-11-31.jpg"), pattern)
        assert result is None

    def test_month_13_returns_none(self):
        """Month 13 is invalid and returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2026-13-01.jpg"), pattern)
        assert result is None

    def test_day_zero_returns_none(self):
        """Day 0 is invalid and returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2026-01-00.jpg"), pattern)
        assert result is None

    def test_month_zero_returns_none(self):
        """Month 0 is invalid and returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2026-00-15.jpg"), pattern)
        assert result is None

    def test_april_31_returns_none(self):
        """Apr 31 does not exist and returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2025-04-31.jpg"), pattern)
        assert result is None


# ===========================================================================
# extract_date_from_path — leap year edge cases
# ===========================================================================


class TestExtractDateFromPathLeapYear:
    def test_valid_leap_year_feb_29_returns_date(self):
        """Feb 29 on a valid leap year (2000) is extracted correctly."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2000-02-29.jpg"), pattern)
        assert result == date(2000, 2, 29)

    def test_invalid_leap_year_feb_29_returns_none(self):
        """Feb 29 on a non-leap year (2001) returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2001-02-29.jpg"), pattern)
        assert result is None

    def test_divisible_by_100_not_400_not_leap(self):
        """Year 1900 (divisible by 100, not by 400) is not a leap year."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/1900-02-29.jpg"), pattern)
        assert result is None

    def test_divisible_by_400_is_leap(self):
        """Year 2400 (divisible by 400) is a leap year and Feb 29 is valid."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/2400-02-29.jpg"), pattern)
        assert result == date(2400, 2, 29)


# ===========================================================================
# extract_date_from_path — no match returns None
# ===========================================================================


class TestExtractDateFromPathNoMatch:
    def test_no_date_in_path_returns_none(self):
        """Path with no date-like substring returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/photos/vacation/beach.jpg"), pattern)
        assert result is None

    def test_wrong_separator_returns_none(self):
        """Path with date using wrong separator does not match."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/photos/20260117.jpg"), pattern)
        assert result is None

    def test_partial_date_in_path_returns_none(self):
        """Path containing only year and month (no day) returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/photos/2026-01.jpg"), pattern)
        assert result is None

    def test_empty_filename_returns_none(self):
        """Path consisting of a bare root returns None."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/"), pattern)
        assert result is None

    def test_date_with_wrong_year_digit_count_returns_none(self):
        """3-digit year does not satisfy the 4-digit {YYYY} requirement."""
        pattern = compile_pattern("{YYYY}-{MM}-{DD}")
        result = extract_date_from_path(Path("/pics/202-01-17.jpg"), pattern)
        assert result is None
