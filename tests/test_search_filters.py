"""Tests for individual filter helpers in search_filters.py (GPS, date, filename)."""

from datetime import datetime

import pytest

from photoaident.core.date_range import DateRange
from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search import search_images
from photoaident.core.search_filters import (
    _filename_filter_clauses,
    _filename_subquery,
    _find_images_by_gps_bbox,
)
from photoaident.db.vector_store import VectorStore
from tests.search_helpers import _add_image_with_metadata

# ---------------------------------------------------------------------------
# GPS filter
# ---------------------------------------------------------------------------


def test_find_images_by_gps_bbox(search_db):
    """Images inside the bounding box are returned; outside and no-GPS are excluded."""
    img1_id = _add_image_with_metadata(
        search_db, "/img1.jpg", "h1", gps_lat=52.52, gps_lon=13.40
    )
    _add_image_with_metadata(search_db, "/img2.jpg", "h2", gps_lat=51.50, gps_lon=-0.12)
    _add_image_with_metadata(search_db, "/img3.jpg", "h3")  # no GPS

    bbox = GpsBoundingBox(south=47.0, west=5.0, north=55.0, east=15.0)
    results = _find_images_by_gps_bbox(search_db, bbox)

    assert results == [img1_id]


def test_find_images_by_gps_bbox_antimeridian(search_db):
    """Bounding boxes that cross the antimeridian include points on both sides."""
    img1_id = _add_image_with_metadata(
        search_db, "/img1.jpg", "h1", gps_lat=-18.0, gps_lon=179.0
    )
    img2_id = _add_image_with_metadata(
        search_db, "/img2.jpg", "h2", gps_lat=-18.0, gps_lon=-179.0
    )
    _add_image_with_metadata(
        search_db, "/img3.jpg", "h3", gps_lat=51.50, gps_lon=-0.12
    )  # outside box

    bbox = GpsBoundingBox(south=-20.0, west=170.0, north=-10.0, east=-170.0)
    results = _find_images_by_gps_bbox(search_db, bbox)

    assert len(results) == 2
    assert set(results) == {img1_id, img2_id}


def test_search_images_gps_only(search_db, tmp_path):
    """search_images filters by GPS when no person_ids are provided."""
    img1_id = _add_image_with_metadata(
        search_db, "/img1.jpg", "h1", gps_lat=52.52, gps_lon=13.40
    )
    _add_image_with_metadata(search_db, "/img2.jpg", "h2", gps_lat=40.0, gps_lon=10.0)

    bbox = GpsBoundingBox(south=52.0, west=13.0, north=53.0, east=14.0)
    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=bbox,
        date_range=None,
    )

    assert len(results) == 1
    assert results[0].image_id == img1_id


# ---------------------------------------------------------------------------
# Date filter
# ---------------------------------------------------------------------------


def test_date_only_filter_includes_images_in_range(search_db, tmp_path):
    """Images with taken_at within the date range are returned."""
    _add_image_with_metadata(
        search_db,
        file_path="/in_range.jpg",
        file_hash="in",
        taken_at=datetime(2021, 6, 15),
    )
    _add_image_with_metadata(
        search_db,
        file_path="/out_range.jpg",
        file_hash="out",
        taken_at=datetime(2019, 3, 1),
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=DateRange(start_year=2020, end_year=2022),
    )

    assert len(results) == 1
    assert results[0].file_path == "/in_range.jpg"


def test_date_only_filter_excludes_null_taken_at(search_db, tmp_path):
    """Images with NULL taken_at are excluded from date range results."""
    _add_image_with_metadata(
        search_db, file_path="/no_date.jpg", file_hash="nodate", taken_at=None
    )
    _add_image_with_metadata(
        search_db,
        file_path="/dated.jpg",
        file_hash="dated",
        taken_at=datetime(2021, 1, 1),
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=DateRange(start_year=2020, end_year=2022),
    )

    paths = {r.file_path for r in results}
    assert "/no_date.jpg" not in paths
    assert "/dated.jpg" in paths


def test_date_filter_open_ended_no_start(search_db, tmp_path):
    """Open-ended range (no start) includes everything up to end date."""
    _add_image_with_metadata(
        search_db, file_path="/old.jpg", file_hash="old", taken_at=datetime(2000, 1, 1)
    )
    _add_image_with_metadata(
        search_db, file_path="/new.jpg", file_hash="new", taken_at=datetime(2025, 1, 1)
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=DateRange(end_year=2020),
    )

    paths = {r.file_path for r in results}
    assert "/old.jpg" in paths
    assert "/new.jpg" not in paths


def test_date_filter_open_ended_no_end(search_db, tmp_path):
    """Open-ended range (no end) includes everything from start date onward."""
    _add_image_with_metadata(
        search_db, file_path="/old.jpg", file_hash="old", taken_at=datetime(2000, 1, 1)
    )
    _add_image_with_metadata(
        search_db, file_path="/new.jpg", file_hash="new", taken_at=datetime(2025, 1, 1)
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=DateRange(start_year=2020),
    )

    paths = {r.file_path for r in results}
    assert "/new.jpg" in paths
    assert "/old.jpg" not in paths


def test_date_filter_with_month_precision(search_db, tmp_path):
    """Month-level precision correctly includes/excludes boundary images."""
    _add_image_with_metadata(
        search_db,
        file_path="/march.jpg",
        file_hash="mar",
        taken_at=datetime(2021, 3, 15),
    )
    _add_image_with_metadata(
        search_db, file_path="/july.jpg", file_hash="jul", taken_at=datetime(2021, 7, 1)
    )
    _add_image_with_metadata(
        search_db,
        file_path="/december.jpg",
        file_hash="dec",
        taken_at=datetime(2021, 12, 31),
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=DateRange(
            start_year=2021, start_month=4, end_year=2021, end_month=9
        ),
    )

    paths = {r.file_path for r in results}
    assert "/march.jpg" not in paths
    assert "/july.jpg" in paths
    assert "/december.jpg" not in paths


# ---------------------------------------------------------------------------
# Filename filter
# ---------------------------------------------------------------------------


def test_search_by_filename_substring(search_db, tmp_path):
    """Images whose path contains the query substring are returned."""
    _add_image_with_metadata(
        search_db, file_path="/photos/New York/best_images/foo.jpg", file_hash="ny"
    )
    _add_image_with_metadata(
        search_db, file_path="/photos/Paris/bar.jpg", file_hash="paris"
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="New York",
    )

    assert len(results) == 1
    assert "New York" in results[0].file_path


def test_search_by_filename_case_insensitive(search_db, tmp_path):
    """Filename search is case-insensitive."""
    _add_image_with_metadata(
        search_db, file_path="/photos/New York/foo.jpg", file_hash="ny2"
    )
    _add_image_with_metadata(
        search_db, file_path="/photos/Paris/bar.jpg", file_hash="paris2"
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="yoRK",
    )

    assert len(results) == 1
    assert "New York" in results[0].file_path


def test_search_by_filename_no_match(search_db, tmp_path):
    """No results returned when query matches nothing."""
    _add_image_with_metadata(
        search_db, file_path="/photos/London/baz.jpg", file_hash="lon"
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="Tokyo",
    )

    assert results == []


def test_search_by_filename_only(search_db, tmp_path):
    """Filename filter works as a standalone filter with no other criteria."""
    _add_image_with_metadata(
        search_db, file_path="/family/summer_2019/picnic.jpg", file_hash="summer"
    )
    _add_image_with_metadata(
        search_db, file_path="/family/winter_2019/snow.jpg", file_hash="winter"
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="summer",
    )

    assert len(results) == 1
    assert "summer" in results[0].file_path


def test_search_empty_filename_ignored(search_db, tmp_path):
    """filename_query=None is not a filter; empty result when no other filters set."""
    _add_image_with_metadata(search_db, file_path="/some/image.jpg", file_hash="img1")

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query=None,
    )

    assert results == []


def test_search_by_filename_multiple_keywords_all_must_match(search_db, tmp_path):
    """All space-separated tokens must appear in the path (AND logic)."""
    _add_image_with_metadata(
        search_db,
        file_path="/photos/New York/summer_2019/picnic.jpg",
        file_hash="ny_summer",
    )
    _add_image_with_metadata(
        search_db,
        file_path="/photos/New York/winter_2020/snow.jpg",
        file_hash="ny_winter",
    )
    _add_image_with_metadata(
        search_db, file_path="/photos/Paris/summer_2019/café.jpg", file_hash="paris"
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="New York summer",
    )

    assert len(results) == 1
    assert "New York" in results[0].file_path
    assert "summer" in results[0].file_path


def test_search_by_filename_multiple_keywords_no_match(search_db, tmp_path):
    """No result when one of the tokens is absent from all paths."""
    _add_image_with_metadata(
        search_db, file_path="/photos/New York/foo.jpg", file_hash="ny3"
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="New York Tokyo",
    )

    assert results == []


def test_search_by_filename_multiple_keywords_case_insensitive(search_db, tmp_path):
    """Multi-keyword search is case-insensitive per token."""
    _add_image_with_metadata(
        search_db,
        file_path="/Photos/New York/Best_Images/foo.jpg",
        file_hash="ny4",
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="new YORK best",
    )

    assert len(results) == 1


# ---------------------------------------------------------------------------
# _filename_subquery — empty / whitespace guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty", ["", "   ", "\t", "\n"])
def test_filename_subquery_empty_matches_nothing(search_db, empty):
    """An empty or whitespace-only query produces a subquery that matches no rows."""
    _add_image_with_metadata(search_db, "/photos/vacation.jpg", "h1")

    with search_db() as session:
        result = list(session.scalars(_filename_subquery(empty)).all())

    assert result == [], f"Expected no matches for query {empty!r}, got {result}"


# ---------------------------------------------------------------------------
# _filename_subquery — LIKE metacharacter escaping
# ---------------------------------------------------------------------------


def test_filename_subquery_percent_is_literal(search_db):
    """A '%' in the query is treated as a literal character, not a LIKE wildcard."""
    literal_id = _add_image_with_metadata(search_db, "/photos/100%_crop.jpg", "h1")
    _add_image_with_metadata(search_db, "/photos/vacation.jpg", "h2")

    with search_db() as session:
        # "100%" must match the first file but NOT act as a wildcard matching everything
        result = list(session.scalars(_filename_subquery("100%")).all())

    assert result == [literal_id]


def test_filename_subquery_underscore_is_literal(search_db):
    """An '_' in the query is treated as a literal character, not a LIKE wildcard."""
    literal_id = _add_image_with_metadata(search_db, "/photos/my_photo.jpg", "h1")
    _add_image_with_metadata(search_db, "/photos/myphoto.jpg", "h2")

    with search_db() as session:
        # "my_photo" should match only the file with the literal underscore
        result = list(session.scalars(_filename_subquery("my_photo")).all())

    assert result == [literal_id]


def test_filename_subquery_percent_does_not_match_all(search_db):
    """A bare '%' query must not match every row (wildcard semantics suppressed)."""
    _add_image_with_metadata(search_db, "/photos/beach.jpg", "h1")
    _add_image_with_metadata(search_db, "/photos/mountain.jpg", "h2")

    with search_db() as session:
        result = list(session.scalars(_filename_subquery("%")).all())

    # "%" contains no literal "%" characters in the DB paths → no match
    assert result == []


# ---------------------------------------------------------------------------
# _filename_filter_clauses — unit tests (no DB round-trip needed)
# ---------------------------------------------------------------------------


def test_filename_filter_clauses_empty_returns_empty_list():
    """An empty query returns no clauses (caller treats this as 'no filter')."""
    assert _filename_filter_clauses("") == []


@pytest.mark.parametrize("whitespace", ["   ", "\t", "\n"])
def test_filename_filter_clauses_whitespace_returns_empty_list(whitespace):
    assert _filename_filter_clauses(whitespace) == []


def test_filename_filter_clauses_single_token_returns_one_clause():
    clauses = _filename_filter_clauses("vacation")
    assert len(clauses) == 1


def test_filename_filter_clauses_multi_token_returns_one_clause_per_token():
    clauses = _filename_filter_clauses("2024 vacation beach")
    assert len(clauses) == 3


# ---------------------------------------------------------------------------
# _search_by_metadata_only — filename applied inline (no self-subquery)
# ---------------------------------------------------------------------------


def test_search_images_filename_only_matches_path(search_db, tmp_path):
    """Filename-only search returns images whose file_path contains the token."""
    _add_image_with_metadata(search_db, "/photos/vacation_2024.jpg", "h1")
    _add_image_with_metadata(search_db, "/photos/birthday_2024.jpg", "h2")

    results = search_images(
        thumbs_dir=tmp_path,
        session_factory=search_db,
        vector_store=VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="vacation",
    )

    assert len(results) == 1
    assert "vacation" in results[0].file_path


def test_search_images_filename_multi_token_and_logic(search_db, tmp_path):
    """All tokens must match — images missing any token are excluded."""
    _add_image_with_metadata(search_db, "/photos/2024/vacation.jpg", "h1")
    _add_image_with_metadata(search_db, "/photos/2023/vacation.jpg", "h2")
    _add_image_with_metadata(search_db, "/photos/2024/birthday.jpg", "h3")

    results = search_images(
        thumbs_dir=tmp_path,
        session_factory=search_db,
        vector_store=VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="2024 vacation",
    )

    assert len(results) == 1
    assert "2024" in results[0].file_path
    assert "vacation" in results[0].file_path


def test_search_images_filename_metachar_percent_is_literal(search_db, tmp_path):
    """A '%' in the filename query matches literally, not as a LIKE wildcard."""
    literal_id = _add_image_with_metadata(search_db, "/photos/100%_crop.jpg", "h1")
    _add_image_with_metadata(search_db, "/photos/vacation.jpg", "h2")

    results = search_images(
        thumbs_dir=tmp_path,
        session_factory=search_db,
        vector_store=VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query="100%",
    )

    assert len(results) == 1
    assert results[0].image_id == literal_id
