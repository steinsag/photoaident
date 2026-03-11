"""Orchestration tests for search_images(): multi-filter combinations and ranking."""

from datetime import datetime

import numpy as np

from photoaident.core.date_range import DateRange
from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search import search_images
from photoaident.db.database import (
    Face,
    FaceState,
    Image,
    ImageMetadata,
    TakenAtSource,
)
from photoaident.db.vector_store import VectorStore
from tests.search_helpers import (
    _add_face_for_image,
    _add_identified_face,
    _add_image_with_metadata,
    _add_person_cluster,
    _rand_norm_emb,
)


def test_search_images_empty_input_returns_empty(search_db, vector_store, tmp_path):
    """search_images returns empty if no filters are provided."""
    results = search_images(
        tmp_path, search_db, vector_store, person_ids=[], gps_bbox=None, date_range=None
    )
    assert results == []


def test_no_filters_returns_empty(search_db, vector_store, tmp_path):
    """search_images with no filters returns empty even when images exist."""
    _add_image_with_metadata(
        search_db, file_path="/img.jpg", file_hash="h", taken_at=datetime(2021, 1, 1)
    )
    results = search_images(
        tmp_path, search_db, vector_store, person_ids=[], gps_bbox=None, date_range=None
    )
    assert results == []


def test_search_images_multiple_persons(search_db, vector_store, tmp_path):
    """search_images requires ALL persons to be present (AND logic)."""
    p1_id, c1_id = _add_person_cluster(search_db)
    p2_id, c2_id = _add_person_cluster(search_db)

    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0

    # Image with both persons
    with search_db() as session:
        img_both = Image(file_path="/both.jpg", file_size=100, file_hash="both")
        session.add(img_both)
        session.flush()
        both_id = img_both.id
        session.add_all(
            [
                Face(
                    image_id=both_id,
                    faiss_id=vector_store.add(emb1),
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=50,
                    bbox_h=50,
                    detection_confidence=0.9,
                    person_id=p1_id,
                    cluster_id=c1_id,
                    state=FaceState.IDENTIFIED,
                    model_version="test",
                ),
                Face(
                    image_id=both_id,
                    faiss_id=vector_store.add(emb2),
                    bbox_x=60,
                    bbox_y=0,
                    bbox_w=50,
                    bbox_h=50,
                    detection_confidence=0.9,
                    person_id=p2_id,
                    cluster_id=c2_id,
                    state=FaceState.IDENTIFIED,
                    model_version="test",
                ),
            ]
        )
        session.commit()

    # Image with only p1
    _add_identified_face(search_db, vector_store, p1_id, c1_id, emb1)

    assert (
        search_images(
            tmp_path,
            search_db,
            vector_store,
            person_ids=[],
            gps_bbox=None,
            date_range=None,
        )
        == []
    )

    results_p1 = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[p1_id],
        gps_bbox=None,
        date_range=None,
    )
    assert len(results_p1) == 2

    results_both = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[p1_id, p2_id],
        gps_bbox=None,
        date_range=None,
    )
    assert len(results_both) == 1
    assert results_both[0].image_id == both_id

    p3_id, _ = _add_person_cluster(search_db)
    assert (
        search_images(
            tmp_path,
            search_db,
            vector_store,
            person_ids=[p1_id, p3_id],
            gps_bbox=None,
            date_range=None,
        )
        == []
    )


def test_search_images_ranking(search_db, vector_store, tmp_path):
    """search_images ranks results by minimum score across multiple persons."""
    p1_id, c1_id = _add_person_cluster(search_db)
    p2_id, c2_id = _add_person_cluster(search_db)

    emb_exact = np.zeros(512, dtype=np.float32)
    emb_exact[0] = 1.0

    with search_db() as session:
        img1 = Image(file_path="/img1.jpg", file_size=100, file_hash="h1")
        img2 = Image(file_path="/img2.jpg", file_size=100, file_hash="h2")
        session.add_all([img1, img2])
        session.flush()

        v09 = emb_exact.copy()
        v09[1] = 0.435  # approx sqrt(1 - 0.9²)
        v09 /= np.linalg.norm(v09)

        v08 = emb_exact.copy()
        v08[1] = 0.6
        v08 /= np.linalg.norm(v08)

        session.add_all(
            [
                Face(
                    image_id=img1.id,
                    faiss_id=vector_store.add(emb_exact),
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=100,
                    bbox_h=100,
                    detection_confidence=0.9,
                    person_id=p1_id,
                    cluster_id=c1_id,
                    state=FaceState.IDENTIFIED,
                    model_version="test",
                ),
                Face(
                    image_id=img1.id,
                    faiss_id=vector_store.add(v09),
                    bbox_x=110,
                    bbox_y=0,
                    bbox_w=100,
                    bbox_h=100,
                    detection_confidence=0.9,
                    person_id=p2_id,
                    cluster_id=c2_id,
                    state=FaceState.IDENTIFIED,
                    model_version="test",
                ),
                Face(
                    image_id=img2.id,
                    faiss_id=vector_store.add(v08),
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=100,
                    bbox_h=100,
                    detection_confidence=0.9,
                    person_id=p1_id,
                    cluster_id=c1_id,
                    state=FaceState.IDENTIFIED,
                    model_version="test",
                ),
                Face(
                    image_id=img2.id,
                    faiss_id=vector_store.add(v08),
                    bbox_x=110,
                    bbox_y=0,
                    bbox_w=100,
                    bbox_h=100,
                    detection_confidence=0.9,
                    person_id=p2_id,
                    cluster_id=c2_id,
                    state=FaceState.IDENTIFIED,
                    model_version="test",
                ),
            ]
        )
        session.commit()
        img1_id, img2_id = img1.id, img2.id

    results = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[p1_id, p2_id],
        gps_bbox=None,
        date_range=None,
    )
    assert len(results) == 2
    assert results[0].image_id == img1_id
    assert results[1].image_id == img2_id


# ---------------------------------------------------------------------------
# Person + GPS
# ---------------------------------------------------------------------------


def test_search_images_person_and_gps_intersection(search_db, vector_store, tmp_path):
    """search_images intersects person and GPS filters."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _rand_norm_emb()

    img1_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )
    with search_db() as session:
        session.add(
            ImageMetadata(
                image_id=img1_id,
                gps_lat=50.0,
                gps_lon=10.0,
                taken_at_source=TakenAtSource.FILESYSTEM,
                width=100,
                height=100,
            )
        )
        session.commit()

    img2_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )
    with search_db() as session:
        session.add(
            ImageMetadata(
                image_id=img2_id,
                gps_lat=20.0,
                gps_lon=20.0,
                taken_at_source=TakenAtSource.FILESYSTEM,
                width=100,
                height=100,
            )
        )
        session.commit()

    bbox = GpsBoundingBox(south=45.0, west=5.0, north=55.0, east=15.0)
    results = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[person_id],
        gps_bbox=bbox,
        date_range=None,
    )

    assert len(results) == 1
    assert results[0].image_id == img1_id


# ---------------------------------------------------------------------------
# Person + date
# ---------------------------------------------------------------------------


def test_date_and_person_intersection(search_db, vector_store, tmp_path):
    """Date filter is applied together with person filter (AND intersection)."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _rand_norm_emb()

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

    results = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[person_id],
        gps_bbox=None,
        date_range=DateRange(start_year=2020, end_year=2022),
    )

    result_ids = {r.image_id for r in results}
    assert img1_id in result_ids
    assert img2_id not in result_ids


# ---------------------------------------------------------------------------
# GPS + date
# ---------------------------------------------------------------------------


def test_date_and_gps_intersection(search_db, tmp_path):
    """Date filter and GPS filter both apply (AND intersection)."""
    _add_image_with_metadata(
        search_db,
        file_path="/both.jpg",
        file_hash="both",
        taken_at=datetime(2021, 6, 1),
        gps_lat=52.0,
        gps_lon=13.0,
    )
    _add_image_with_metadata(
        search_db,
        file_path="/date_only.jpg",
        file_hash="dateonly",
        taken_at=datetime(2021, 6, 1),
        gps_lat=10.0,
        gps_lon=10.0,
    )
    _add_image_with_metadata(
        search_db,
        file_path="/gps_only.jpg",
        file_hash="gpsonly",
        taken_at=datetime(2015, 1, 1),
        gps_lat=52.0,
        gps_lon=13.0,
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=GpsBoundingBox(south=50.0, west=10.0, north=55.0, east=16.0),
        date_range=DateRange(start_year=2020, end_year=2022),
    )

    paths = {r.file_path for r in results}
    assert "/both.jpg" in paths
    assert "/date_only.jpg" not in paths
    assert "/gps_only.jpg" not in paths


# ---------------------------------------------------------------------------
# Person + GPS + date
# ---------------------------------------------------------------------------


def test_all_three_filters_combined(search_db, vector_store, tmp_path):
    """Person + GPS + date all combined in one search."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0

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

    results = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[person_id],
        gps_bbox=GpsBoundingBox(south=50.0, west=10.0, north=55.0, east=16.0),
        date_range=DateRange(start_year=2020, end_year=2022),
    )

    result_ids = {r.image_id for r in results}
    assert img_match_id in result_ids
    assert img_wrong_date_id not in result_ids


# ---------------------------------------------------------------------------
# Filename + other filters
# ---------------------------------------------------------------------------


def test_search_by_filename_combined_with_date(search_db, tmp_path):
    """Filename filter ANDs with date range filter."""
    _add_image_with_metadata(
        search_db,
        file_path="/vacation/Rome/img.jpg",
        file_hash="rome_in",
        taken_at=datetime(2021, 6, 1),
    )
    _add_image_with_metadata(
        search_db,
        file_path="/vacation/Rome/old.jpg",
        file_hash="rome_out",
        taken_at=datetime(2010, 1, 1),
    )
    _add_image_with_metadata(
        search_db,
        file_path="/vacation/Berlin/img.jpg",
        file_hash="berlin_in",
        taken_at=datetime(2021, 6, 1),
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=None,
        date_range=DateRange(start_year=2020, end_year=2022),
        filename_query="Rome",
    )

    assert len(results) == 1
    assert results[0].file_path == "/vacation/Rome/img.jpg"


def test_search_by_filename_combined_with_gps(search_db, tmp_path):
    """Filename filter ANDs with GPS bounding box filter."""
    _add_image_with_metadata(
        search_db,
        file_path="/trip/Barcelona/beach.jpg",
        file_hash="bcn_in",
        gps_lat=41.4,
        gps_lon=2.2,
    )
    _add_image_with_metadata(
        search_db,
        file_path="/trip/Barcelona/mountain.jpg",
        file_hash="bcn_out",
        gps_lat=10.0,
        gps_lon=10.0,
    )
    _add_image_with_metadata(
        search_db,
        file_path="/trip/Madrid/plaza.jpg",
        file_hash="mad_in",
        gps_lat=41.4,
        gps_lon=2.2,
    )

    results = search_images(
        tmp_path,
        search_db,
        VectorStore(),
        person_ids=[],
        gps_bbox=GpsBoundingBox(south=40.0, west=1.0, north=43.0, east=4.0),
        date_range=None,
        filename_query="Barcelona",
    )

    assert len(results) == 1
    assert results[0].file_path == "/trip/Barcelona/beach.jpg"


def test_search_by_filename_combined_with_person(search_db, vector_store, tmp_path):
    """Filename filter ANDs with person filter in person-based search path."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _rand_norm_emb()

    img1_id = _add_face_for_image(
        search_db, vector_store, person_id, cluster_id, "/New York/photo.jpg", emb
    )
    img2_id = _add_face_for_image(
        search_db, vector_store, person_id, cluster_id, "/London/photo.jpg", emb
    )

    results = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[person_id],
        gps_bbox=None,
        date_range=None,
        filename_query="york",
    )

    result_ids = {r.image_id for r in results}
    assert img1_id in result_ids
    assert img2_id not in result_ids
