"""Tests for date-range filtering in search_images."""

from datetime import datetime

import numpy as np

from photoaident.core.date_range import DateRange
from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search import search_images
from photoaident.db.database import (
    ImageMetadata,
    TakenAtSource,
)
from photoaident.db.vector_store import VectorStore
from tests.search_helpers import (
    _add_identified_face,
    _add_image_with_metadata,
    _add_person_cluster,
    _rand_norm_emb,
)


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

    date_range = DateRange(start_year=2020, end_year=2022)
    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=date_range,
    )

    assert len(results) == 1
    assert results[0].file_path == "/in_range.jpg"


def test_date_only_filter_excludes_null_taken_at(search_db, tmp_path):
    """Images with NULL taken_at are excluded from date range results."""
    _add_image_with_metadata(
        search_db,
        file_path="/no_date.jpg",
        file_hash="nodate",
        taken_at=None,
    )
    _add_image_with_metadata(
        search_db,
        file_path="/dated.jpg",
        file_hash="dated",
        taken_at=datetime(2021, 1, 1),
    )

    date_range = DateRange(start_year=2020, end_year=2022)
    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=date_range,
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

    date_range = DateRange(end_year=2020)
    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=date_range,
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

    date_range = DateRange(start_year=2020)
    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=date_range,
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

    date_range = DateRange(start_year=2021, start_month=4, end_year=2021, end_month=9)
    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=date_range,
    )

    paths = {r.file_path for r in results}
    assert "/march.jpg" not in paths
    assert "/july.jpg" in paths
    assert "/december.jpg" not in paths


def test_date_and_person_intersection(search_db, vector_store, tmp_path):
    """Date filter is applied together with person filter (AND intersection)."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _rand_norm_emb()

    # Image 1: person present, in date range
    img1_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )
    with search_db() as session:
        session.add(
            ImageMetadata(
                image_id=img1_id,
                taken_at=datetime(2021, 5, 1),
                taken_at_source=TakenAtSource.FILESYSTEM,
                width=100,
                height=100,
            )
        )
        session.commit()

    # Image 2: person present, outside date range
    img2_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )
    with search_db() as session:
        session.add(
            ImageMetadata(
                image_id=img2_id,
                taken_at=datetime(2018, 1, 1),
                taken_at_source=TakenAtSource.FILESYSTEM,
                width=100,
                height=100,
            )
        )
        session.commit()

    date_range = DateRange(start_year=2020, end_year=2022)
    results = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[person_id],
        gps_bbox=None,
        date_range=date_range,
    )

    result_ids = {r.image_id for r in results}
    assert img1_id in result_ids
    assert img2_id not in result_ids


def test_date_and_gps_intersection(search_db, tmp_path):
    """Date filter and GPS filter both apply (AND intersection)."""
    # Image 1: in date range AND in GPS box
    _add_image_with_metadata(
        search_db,
        file_path="/both.jpg",
        file_hash="both",
        taken_at=datetime(2021, 6, 1),
        gps_lat=52.0,
        gps_lon=13.0,
    )
    # Image 2: in date range, outside GPS box
    _add_image_with_metadata(
        search_db,
        file_path="/date_only.jpg",
        file_hash="dateonly",
        taken_at=datetime(2021, 6, 1),
        gps_lat=10.0,
        gps_lon=10.0,
    )
    # Image 3: in GPS box, outside date range
    _add_image_with_metadata(
        search_db,
        file_path="/gps_only.jpg",
        file_hash="gpsonly",
        taken_at=datetime(2015, 1, 1),
        gps_lat=52.0,
        gps_lon=13.0,
    )

    bbox = GpsBoundingBox(south=50.0, west=10.0, north=55.0, east=16.0)
    date_range = DateRange(start_year=2020, end_year=2022)
    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=bbox,
        date_range=date_range,
    )

    paths = {r.file_path for r in results}
    assert "/both.jpg" in paths
    assert "/date_only.jpg" not in paths
    assert "/gps_only.jpg" not in paths


def test_all_three_filters_combined(search_db, vector_store, tmp_path):
    """Person + GPS + date all combined in one search."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0

    # The matching image: person + GPS + date
    img_match_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )
    with search_db() as session:
        session.add(
            ImageMetadata(
                image_id=img_match_id,
                taken_at=datetime(2021, 6, 1),
                gps_lat=52.0,
                gps_lon=13.0,
                taken_at_source=TakenAtSource.FILESYSTEM,
                width=100,
                height=100,
            )
        )
        session.commit()

    # A non-matching image: wrong date
    img_wrong_date_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )
    with search_db() as session:
        session.add(
            ImageMetadata(
                image_id=img_wrong_date_id,
                taken_at=datetime(2015, 1, 1),
                gps_lat=52.0,
                gps_lon=13.0,
                taken_at_source=TakenAtSource.FILESYSTEM,
                width=100,
                height=100,
            )
        )
        session.commit()

    bbox = GpsBoundingBox(south=50.0, west=10.0, north=55.0, east=16.0)
    date_range = DateRange(start_year=2020, end_year=2022)
    results = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[person_id],
        gps_bbox=bbox,
        date_range=date_range,
    )

    result_ids = {r.image_id for r in results}
    assert img_match_id in result_ids
    assert img_wrong_date_id not in result_ids


def test_no_filters_returns_empty(search_db, vector_store, tmp_path):
    """search_images with no filters returns empty list."""
    _add_image_with_metadata(
        search_db, file_path="/img.jpg", file_hash="h", taken_at=datetime(2021, 1, 1)
    )
    results = search_images(
        tmp_path, search_db, vector_store, person_ids=[], gps_bbox=None, date_range=None
    )
    assert results == []
