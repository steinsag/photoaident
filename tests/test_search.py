"""Combined search tests: empty input, person+GPS, multi-person, ranking."""

import numpy as np

from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search import search_images
from photoaident.db.database import (
    Face,
    FaceState,
    Image,
    ImageMetadata,
    TakenAtSource,
)
from tests.search_helpers import (
    _add_identified_face,
    _add_person_cluster,
    _rand_norm_emb,
)


def test_search_images_empty_input_returns_empty(search_db, vector_store, tmp_path):
    """search_images returns empty if no filters are provided."""
    results = search_images(
        tmp_path, search_db, vector_store, person_ids=[], gps_bbox=None
    )
    assert results == []


def test_search_images_intersection(search_db, vector_store, tmp_path):
    """search_images intersects person and GPS filters."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _rand_norm_emb()

    # Image 1: has person, inside GPS
    img1_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )
    with search_db() as session:
        meta1 = ImageMetadata(
            image_id=img1_id,
            gps_lat=50.0,
            gps_lon=10.0,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta1)
        session.commit()

    # Image 2: has person, outside GPS
    img2_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )
    with search_db() as session:
        meta2 = ImageMetadata(
            image_id=img2_id,
            gps_lat=20.0,
            gps_lon=20.0,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta2)
        session.commit()

    bbox = GpsBoundingBox(south=45.0, west=5.0, north=55.0, east=15.0)
    results = search_images(
        tmp_path, search_db, vector_store, person_ids=[person_id], gps_bbox=bbox
    )

    assert len(results) == 1
    assert results[0].image_id == img1_id


def test_search_images_multiple_persons(search_db, vector_store, tmp_path):
    """search_images requires ALL persons to be present (AND logic)."""
    p1_id, c1_id = _add_person_cluster(search_db)
    p2_id, c2_id = _add_person_cluster(search_db)

    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0

    # Image with both
    with search_db() as session:
        img_both = Image(file_path="/both.jpg", file_size=100, file_hash="both")
        session.add(img_both)
        session.flush()
        both_id = img_both.id
        f1 = Face(
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
        )
        f2 = Face(
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
        )
        session.add_all([f1, f2])
        session.commit()

    # Image with only p1
    _add_identified_face(search_db, vector_store, p1_id, c1_id, emb1)

    # Empty person_ids list
    results_empty = search_images(
        tmp_path, search_db, vector_store, person_ids=[], gps_bbox=None
    )
    assert results_empty == []

    # One person (p1)
    results_p1 = search_images(
        tmp_path, search_db, vector_store, person_ids=[p1_id], gps_bbox=None
    )
    assert len(results_p1) == 2  # both.jpg and img1.jpg

    # Intersection of p1 and p2
    results_both = search_images(
        tmp_path, search_db, vector_store, person_ids=[p1_id, p2_id], gps_bbox=None
    )
    assert len(results_both) == 1
    assert results_both[0].image_id == both_id

    # Intersection with no common images
    p3_id, _ = _add_person_cluster(search_db)
    results_none = search_images(
        tmp_path, search_db, vector_store, person_ids=[p1_id, p3_id], gps_bbox=None
    )
    assert results_none == []


def test_search_images_ranking(search_db, vector_store, tmp_path):
    """search_images ranks by minimum score across multiple persons."""
    p1_id, c1_id = _add_person_cluster(search_db)
    p2_id, c2_id = _add_person_cluster(search_db)

    emb_exact = np.zeros(512, dtype=np.float32)
    emb_exact[0] = 1.0

    # Image 1: p1 matched at 1.0, p2 matched at 0.9
    # Image 2: p1 matched at 0.8, p2 matched at 0.8
    # Image 1 min score: 0.9, Image 2 min score: 0.8 -> Image 1 should be first

    with search_db() as session:
        img1 = Image(file_path="/img1.jpg", file_size=100, file_hash="h1")
        img2 = Image(file_path="/img2.jpg", file_size=100, file_hash="h2")
        session.add_all([img1, img2])
        session.flush()

        # Image 1 faces
        session.add(
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
            )
        )
        # Using a vector very close to emb_exact for 0.9 score
        v09 = emb_exact.copy()
        v09[1] = 0.435  # sqrt(1-0.9^2) approx 0.4358
        v09 /= np.linalg.norm(v09)
        session.add(
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
            )
        )

        # Image 2 faces (both 0.8)
        v08 = emb_exact.copy()
        v08[1] = 0.6
        v08 /= np.linalg.norm(v08)
        session.add(
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
            )
        )
        session.add(
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
            )
        )
        session.commit()
        img1_id, img2_id = img1.id, img2.id

    results = search_images(
        tmp_path, search_db, vector_store, person_ids=[p1_id, p2_id], gps_bbox=None
    )
    assert len(results) == 2
    assert results[0].image_id == img1_id
    assert results[1].image_id == img2_id
