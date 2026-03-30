"""Orchestration tests for search_images(): multi-filter combinations and ranking."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from photoaident.core.date_range import DateRange
from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search import SearchResult, SortOrder, search_images, sort_results
from photoaident.db.cluster_means import recompute_cluster_mean
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
    _unit_emb,
    _zero_emb,
)

# ---------------------------------------------------------------------------
# sort_results() unit tests
# ---------------------------------------------------------------------------


def _sr(
    image_id: int,
    file_path: str,
    score: float | None = None,
    taken_at: datetime | None = None,
) -> SearchResult:
    return SearchResult(
        image_id=image_id,
        file_path=file_path,
        thumb_path=Path(file_path),
        score=score,
        taken_at=taken_at,
    )


def test_sort_relevance_desc():
    """RELEVANCE_DESC: highest score first; None scores last."""
    results = [
        _sr(1, "/a.jpg", score=0.5),
        _sr(2, "/b.jpg", score=None),
        _sr(3, "/c.jpg", score=0.9),
        _sr(4, "/d.jpg", score=0.2),
    ]
    sorted_results = sort_results(results, SortOrder.RELEVANCE_DESC)
    ids = [r.image_id for r in sorted_results]
    assert ids == [3, 1, 4, 2]


def test_sort_relevance_asc():
    """RELEVANCE_ASC: lowest score first; None scores last."""
    results = [
        _sr(1, "/a.jpg", score=0.5),
        _sr(2, "/b.jpg", score=None),
        _sr(3, "/c.jpg", score=0.9),
        _sr(4, "/d.jpg", score=0.2),
    ]
    sorted_results = sort_results(results, SortOrder.RELEVANCE_ASC)
    ids = [r.image_id for r in sorted_results]
    assert ids == [4, 1, 3, 2]


def test_sort_taken_at_desc():
    """TAKEN_AT_DESC: newest first; None dates last."""
    dt1 = datetime(2020, 1, 1)
    dt2 = datetime(2022, 6, 15)
    results = [
        _sr(1, "/a.jpg", taken_at=dt1),
        _sr(2, "/b.jpg", taken_at=None),
        _sr(3, "/c.jpg", taken_at=dt2),
    ]
    sorted_results = sort_results(results, SortOrder.TAKEN_AT_DESC)
    ids = [r.image_id for r in sorted_results]
    assert ids == [3, 1, 2]


def test_sort_taken_at_asc():
    """TAKEN_AT_ASC: oldest first; None dates last."""
    dt1 = datetime(2020, 1, 1)
    dt2 = datetime(2022, 6, 15)
    results = [
        _sr(1, "/a.jpg", taken_at=dt1),
        _sr(2, "/b.jpg", taken_at=None),
        _sr(3, "/c.jpg", taken_at=dt2),
    ]
    sorted_results = sort_results(results, SortOrder.TAKEN_AT_ASC)
    ids = [r.image_id for r in sorted_results]
    assert ids == [1, 3, 2]


def test_sort_filename_asc():
    """FILENAME_ASC: alphabetical by path, case-insensitive."""
    results = [
        _sr(1, "/Zebra.jpg"),
        _sr(2, "/apple.jpg"),
        _sr(3, "/Mango.jpg"),
    ]
    sorted_results = sort_results(results, SortOrder.FILENAME_ASC)
    ids = [r.image_id for r in sorted_results]
    assert ids == [2, 3, 1]


def test_sort_filename_desc():
    """FILENAME_DESC: reverse alphabetical by path, case-insensitive."""
    results = [
        _sr(1, "/Zebra.jpg"),
        _sr(2, "/apple.jpg"),
        _sr(3, "/Mango.jpg"),
    ]
    sorted_results = sort_results(results, SortOrder.FILENAME_DESC)
    ids = [r.image_id for r in sorted_results]
    assert ids == [1, 3, 2]


def test_sort_results_does_not_mutate_input():
    """sort_results returns a new list; original is unchanged."""
    results = [_sr(1, "/b.jpg", score=0.5), _sr(2, "/a.jpg", score=0.9)]
    original_ids = [r.image_id for r in results]
    sort_results(results, SortOrder.RELEVANCE_DESC)
    assert [r.image_id for r in results] == original_ids


def test_sort_results_all_none_scores():
    """All None scores: stable result set returned regardless of order."""
    results = [_sr(1, "/a.jpg"), _sr(2, "/b.jpg"), _sr(3, "/c.jpg")]
    for order in (SortOrder.RELEVANCE_DESC, SortOrder.RELEVANCE_ASC):
        sorted_results = sort_results(results, order)
        assert len(sorted_results) == 3


def test_sort_results_empty_list():
    """Empty input returns empty output for all sort orders."""
    for order in SortOrder:
        assert sort_results([], order) == []


# ---------------------------------------------------------------------------
# search_images() tests
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_filename_query_returns_empty(search_db, vector_store, tmp_path, blank):
    """Blank/whitespace filename_query must be treated as no filter → empty result."""
    _add_image_with_metadata(
        search_db,
        file_path="/photos/img.jpg",
        file_hash="h",
        taken_at=datetime(2021, 1, 1),
    )
    results = search_images(
        tmp_path,
        search_db,
        vector_store,
        person_ids=[],
        gps_bbox=None,
        date_range=None,
        filename_query=blank,
    )
    assert results == []


def test_search_images_multiple_persons(search_db, vector_store, tmp_path):
    """search_images requires ALL persons to be present (AND logic)."""
    p1_id, c1_id = _add_person_cluster(search_db)
    p2_id, c2_id = _add_person_cluster(search_db)

    emb1 = _unit_emb(0)
    emb2 = _unit_emb(1)

    # Image with both persons
    with search_db() as session:
        img_both = Image(file_path="/both.jpg", file_size=100, file_hash="both")
        session.add(img_both)
        session.flush()
        both_id = img_both.id
        face_p1 = Face(
            image_id=both_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            person_id=p1_id,
            cluster_id=c1_id,
            state=FaceState.IDENTIFIED,
            model_version="test",
        )
        face_p2 = Face(
            image_id=both_id,
            bbox_x=60,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            person_id=p2_id,
            cluster_id=c2_id,
            state=FaceState.IDENTIFIED,
            model_version="test",
        )
        session.add_all([face_p1, face_p2])
        session.flush()
        vector_store.add(face_p1.id, emb1)
        vector_store.add(face_p2.id, emb2)
        session.commit()

    recompute_cluster_mean(c1_id, search_db, vector_store)
    recompute_cluster_mean(c2_id, search_db, vector_store)

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
    """search_images ranks results by minimum score across multiple persons.

    Fixture design: each cluster has exactly ONE identified face (in img1), so
    the cluster mean equals that face's embedding exactly — making FAISS scores
    fully deterministic.  img2 carries only UNIDENTIFIED faces (excluded from
    mean computation) with a lower-similarity embedding so it ranks second.
    """
    p1_id, c1_id = _add_person_cluster(search_db)
    p2_id, c2_id = _add_person_cluster(search_db)

    # Orthogonal unit vectors — cluster means are exactly these after one face each.
    emb_p1 = _unit_emb(0)
    emb_p2 = _unit_emb(1)

    # 45-degree mix: cosine similarity ~0.707 with both emb_p1 and emb_p2.
    emb_mix = _zero_emb()
    emb_mix[0] = 1.0
    emb_mix[1] = 1.0
    emb_mix /= np.linalg.norm(emb_mix)

    with search_db() as session:
        img1 = Image(file_path="/img1.jpg", file_size=100, file_hash="h1")
        img2 = Image(file_path="/img2.jpg", file_size=100, file_hash="h2")
        session.add_all([img1, img2])
        session.flush()

        # img1: one identified face per person — cluster mean = that embedding.
        face_i1_p1 = Face(
            image_id=img1.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=100,
            bbox_h=100,
            detection_confidence=0.9,
            person_id=p1_id,
            cluster_id=c1_id,
            state=FaceState.IDENTIFIED,
            model_version="test",
        )
        face_i1_p2 = Face(
            image_id=img1.id,
            bbox_x=110,
            bbox_y=0,
            bbox_w=100,
            bbox_h=100,
            detection_confidence=0.9,
            person_id=p2_id,
            cluster_id=c2_id,
            state=FaceState.IDENTIFIED,
            model_version="test",
        )
        # img2: unidentified faces — excluded from cluster mean computation
        # but still found by FAISS search (cosine ~0.707, above threshold).
        face_i2_u1 = Face(
            image_id=img2.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=100,
            bbox_h=100,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        face_i2_u2 = Face(
            image_id=img2.id,
            bbox_x=110,
            bbox_y=0,
            bbox_w=100,
            bbox_h=100,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add_all([face_i1_p1, face_i1_p2, face_i2_u1, face_i2_u2])
        session.flush()
        vector_store.add(face_i1_p1.id, emb_p1)
        vector_store.add(face_i1_p2.id, emb_p2)
        vector_store.add(face_i2_u1.id, emb_mix)
        vector_store.add(face_i2_u2.id, emb_mix)
        session.commit()
        img1_id, img2_id = img1.id, img2.id

    recompute_cluster_mean(c1_id, search_db, vector_store)
    recompute_cluster_mean(c2_id, search_db, vector_store)

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
                taken_at_source=TakenAtSource.EXIF,
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
                taken_at_source=TakenAtSource.EXIF,
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
                taken_at_source=TakenAtSource.EXIF,
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
                taken_at_source=TakenAtSource.EXIF,
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
    emb = _unit_emb(0)

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
                taken_at_source=TakenAtSource.EXIF,
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
                taken_at_source=TakenAtSource.EXIF,
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
