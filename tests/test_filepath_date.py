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


@pytest.mark.parametrize(
    "pattern_str, expected_code",
    [
        ("", PatternErrorCode.EMPTY),
        ("{MM}-{DD}", PatternErrorCode.YEAR_MISSING),
        ("{YYYY}{YYYY}{M}{D}", PatternErrorCode.YEAR_DUPLICATE),
        ("{YYYY}-{MM}-{M}-{DD}", PatternErrorCode.MONTH_CONFLICT),
        ("{YYYY}-{DD}", PatternErrorCode.MONTH_MISSING),
        ("{YYYY}{MM}{MM}{D}", PatternErrorCode.MONTH_MM_DUPLICATE),
        ("{YYYY}{M}{M}{DD}", PatternErrorCode.MONTH_M_DUPLICATE),
        ("{YYYY}-{MM}-{DD}-{D}", PatternErrorCode.DAY_CONFLICT),
        ("{YYYY}-{MM}", PatternErrorCode.DAY_MISSING),
        ("{YYYY}{M}{DD}{DD}", PatternErrorCode.DAY_DD_DUPLICATE),
        ("{YYYY}{MM}{D}{D}", PatternErrorCode.DAY_D_DUPLICATE),
        ("{YYYY}", PatternErrorCode.MONTH_MISSING),
    ],
)
def test_compile_pattern_invalid(pattern_str: str, expected_code: PatternErrorCode):
    """compile_pattern raises PatternValidationError with the expected error code."""
    with pytest.raises(PatternValidationError) as exc_info:
        compile_pattern(pattern_str)
    assert exc_info.value.code is expected_code


def test_compile_pattern_invalid_is_value_error():
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


@pytest.mark.parametrize(
    "path_str",
    [
        "/pics/2026-02-30.jpg",  # Feb 30 does not exist
        "/pics/2026-11-31.jpg",  # Nov 31 does not exist
        "/pics/2026-13-01.jpg",  # month 13 is invalid
        "/pics/2026-01-00.jpg",  # day 0 is invalid
        "/pics/2026-00-15.jpg",  # month 0 is invalid
        "/pics/2025-04-31.jpg",  # Apr 31 does not exist
    ],
)
def test_extract_date_from_path_invalid_dates_return_none(path_str: str):
    """Paths containing impossible calendar dates return None."""
    pattern = compile_pattern("{YYYY}-{MM}-{DD}")
    assert extract_date_from_path(Path(path_str), pattern) is None


# ===========================================================================
# extract_date_from_path — leap year edge cases
# ===========================================================================


@pytest.mark.parametrize(
    "path_str, expected",
    [
        ("/pics/2000-02-29.jpg", date(2000, 2, 29)),  # divisible by 400 → leap
        ("/pics/2001-02-29.jpg", None),  # not a leap year
        ("/pics/1900-02-29.jpg", None),  # divisible by 100 but not 400 → not leap
        ("/pics/2400-02-29.jpg", date(2400, 2, 29)),  # divisible by 400 → leap
    ],
)
def test_extract_date_from_path_leap_year(path_str: str, expected: date | None):
    """Leap year rules are correctly applied when parsing Feb 29 dates."""
    pattern = compile_pattern("{YYYY}-{MM}-{DD}")
    assert extract_date_from_path(Path(path_str), pattern) == expected


# ===========================================================================
# extract_date_from_path — no match returns None
# ===========================================================================


@pytest.mark.parametrize(
    "path_str",
    [
        "/photos/vacation/beach.jpg",  # no date-like substring
        "/photos/20260117.jpg",  # compact date, wrong separator for pattern
        "/photos/2026-01.jpg",  # partial date (year+month only)
        "/",  # bare root, no filename
        "/pics/202-01-17.jpg",  # 3-digit year, not 4-digit {YYYY}
    ],
)
def test_extract_date_from_path_no_match_returns_none(path_str: str):
    """Paths that do not contain a matching date string return None."""
    pattern = compile_pattern("{YYYY}-{MM}-{DD}")
    assert extract_date_from_path(Path(path_str), pattern) is None
