"""Extract taken-at dates from image file paths using configurable patterns.

This module provides a simple placeholder scheme for matching date components
in file paths when EXIF metadata is unavailable.

Supported placeholders:
    {YYYY}  – exactly 4-digit year
    {MM}    – exactly 2-digit month
    {M}     – 1-or-2-digit month
    {DD}    – exactly 2-digit day
    {D}     – 1-or-2-digit day

Example patterns:
    ``{YYYY}-{MM}-{DD}`` matches ``2026-01-17`` anywhere in the path.
    ``{DD}.{M}.{YYYY}`` matches ``17.1.2026``.
    ``PXL{YYYY}{MM}{DD}`` matches ``PXL20260117`` in the filename.
"""

import logging
import re
from datetime import date
from enum import Enum
from pathlib import Path


class PatternErrorCode(Enum):
    """Structured error codes for :class:`PatternValidationError`."""

    EMPTY = "empty"
    YEAR_MISSING = "year_missing"
    YEAR_DUPLICATE = "year_duplicate"
    MONTH_CONFLICT = "month_conflict"
    MONTH_MISSING = "month_missing"
    MONTH_MM_DUPLICATE = "month_mm_duplicate"
    MONTH_M_DUPLICATE = "month_m_duplicate"
    DAY_CONFLICT = "day_conflict"
    DAY_MISSING = "day_missing"
    DAY_DD_DUPLICATE = "day_dd_duplicate"
    DAY_D_DUPLICATE = "day_d_duplicate"


class PatternValidationError(ValueError):
    """Raised when a filepath date pattern fails validation.

    Attributes:
        code: Machine-readable :class:`PatternErrorCode` identifying the
            specific constraint that was violated.
    """

    def __init__(self, code: PatternErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


# Map each placeholder to its named-capture regex fragment.
_PLACEHOLDER_PATTERNS: dict[str, str] = {
    "{YYYY}": r"(?P<year>\d{4})",
    "{MM}": r"(?P<month>\d{2})",
    "{M}": r"(?P<month>\d{1,2})",
    "{DD}": r"(?P<day>\d{2})",
    "{D}": r"(?P<day>\d{1,2})",
}

logger = logging.getLogger(__name__)


def _validate_pattern(pattern: str) -> None:
    """Raise :class:`PatternValidationError` if *pattern* is invalid."""
    if not pattern:
        raise PatternValidationError(
            PatternErrorCode.EMPTY, "Pattern must not be empty."
        )

    # Year: exactly one {YYYY}
    if "{YYYY}" not in pattern:
        raise PatternValidationError(
            PatternErrorCode.YEAR_MISSING,
            "Pattern must contain exactly one {YYYY} placeholder.",
        )
    if pattern.count("{YYYY}") > 1:
        raise PatternValidationError(
            PatternErrorCode.YEAR_DUPLICATE,
            "Pattern must not contain {YYYY} more than once.",
        )

    # Month: pattern must contain exactly one month placeholder, either {MM} or {M}.
    has_mm = "{MM}" in pattern
    has_m = "{M}" in pattern
    if has_mm and has_m:
        raise PatternValidationError(
            PatternErrorCode.MONTH_CONFLICT,
            "Pattern must not contain both {MM} and {M}.",
        )
    if not has_mm and not has_m:
        raise PatternValidationError(
            PatternErrorCode.MONTH_MISSING,
            "Pattern must contain a month placeholder ({MM} or {M}).",
        )
    if has_mm and pattern.count("{MM}") > 1:
        raise PatternValidationError(
            PatternErrorCode.MONTH_MM_DUPLICATE,
            "Pattern must not contain {MM} more than once.",
        )
    if has_m and pattern.count("{M}") > 1:
        raise PatternValidationError(
            PatternErrorCode.MONTH_M_DUPLICATE,
            "Pattern must not contain {M} more than once.",
        )

    # Day: exactly one of {DD} or {D}
    has_dd = "{DD}" in pattern
    has_d = "{D}" in pattern
    if has_dd and has_d:
        raise PatternValidationError(
            PatternErrorCode.DAY_CONFLICT,
            "Pattern must not contain both {DD} and {D}.",
        )
    if not has_dd and not has_d:
        raise PatternValidationError(
            PatternErrorCode.DAY_MISSING,
            "Pattern must contain a day placeholder ({DD} or {D}).",
        )
    if has_dd and pattern.count("{DD}") > 1:
        raise PatternValidationError(
            PatternErrorCode.DAY_DD_DUPLICATE,
            "Pattern must not contain {DD} more than once.",
        )
    if has_d and pattern.count("{D}") > 1:
        raise PatternValidationError(
            PatternErrorCode.DAY_D_DUPLICATE,
            "Pattern must not contain {D} more than once.",
        )


def _build_regex(pattern: str) -> str:
    """Translate a validated placeholder pattern into a raw regex string.

    Scans left-to-right, always picking the longest matching placeholder at
    each position so that ``{M}`` never matches inside ``{MM}``.
    """
    regex_parts: list[str] = []
    remaining = pattern
    while remaining:
        best: tuple[int, str] | None = None
        for ph in _PLACEHOLDER_PATTERNS:
            idx = remaining.find(ph)
            if idx < 0:
                continue
            if (
                best is None
                or idx < best[0]
                or (idx == best[0] and len(ph) > len(best[1]))
            ):
                best = (idx, ph)
        if best is None:
            regex_parts.append(re.escape(remaining))
            break
        idx, ph = best
        regex_parts.append(re.escape(remaining[:idx]))
        regex_parts.append(_PLACEHOLDER_PATTERNS[ph])
        remaining = remaining[idx + len(ph) :]
    return "".join(regex_parts)


def compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a placeholder pattern string into a regex.

    Args:
        pattern: A string containing exactly one ``{YYYY}``, exactly one month
            placeholder (``{MM}`` *or* ``{M}``, not both), and exactly one day
            placeholder (``{DD}`` *or* ``{D}``, not both).  All other
            characters are treated as literals.

    Returns:
        A compiled ``re.Pattern`` with named groups ``year``, ``month``, ``day``.

    Raises:
        ValueError: If the pattern is invalid (missing placeholder, conflicts,
            or empty).
    """
    _validate_pattern(pattern)
    regex = _build_regex(pattern)
    logger.debug("Compiled filepath date pattern: %s", regex)
    return re.compile(regex)


def extract_date_from_path(
    path: Path, compiled_pattern: re.Pattern[str]
) -> date | None:
    """Search a file path string for a date matching the compiled pattern.

    Args:
        path: The image file path to search.
        compiled_pattern: A pattern returned by :func:`compile_pattern`.

    Returns:
        A :class:`datetime.date` if a valid date is found, or ``None`` if the
        pattern does not match or the matched values do not form a valid
        calendar date (e.g. February 30).
    """
    match = compiled_pattern.search(str(path))
    if match is None:
        return None

    try:
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
        return date(year, month, day)
    except (ValueError, IndexError):
        return None
