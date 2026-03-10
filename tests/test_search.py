"""Tests for core.search.find_images_by_person."""

import numpy as np
import pytest
from sqlalchemy import create_engine

from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search import _find_images_by_person, search_images
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    ImageMetadata,
    Person,
    TakenAtSource,
    get_session_factory,
)
from photoaident.db.migrate import apply_migrations
from photoaident.db.vector_store import VectorStore


@pytest.fixture
def search_db(tmp_path):
    """Fresh per-test SQLite DB with migrations applied."""
    db_path = tmp_path / "search.db"
    apply_migrations(f"sqlite:///{db_path}")
    engine = create_engine(f"sqlite:///{db_path}")
    return get_session_factory(engine)


@pytest.fixture
def vs():
    return VectorStore()


_rng = np.random.default_rng(seed=42)


def _rand_norm_emb() -> np.ndarray:
    v = _rng.random(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _add_person_cluster(session_factory) -> tuple[int, int]:
    """Insert a Person + EmbeddingCluster; return (person_id, cluster_id)."""
    with session_factory() as session:
        person = Person(name="Test Person")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(person_id=person.id)
        session.add(cluster)
        session.commit()
        return person.id, cluster.id


def _add_identified_face(
    session_factory,
    vector_store: VectorStore,
    person_id: int,
    cluster_id: int,
    embedding: np.ndarray,
) -> tuple[int, int]:
    """Insert an identified Face; return (image_id, faiss_id)."""
    faiss_id = vector_store.add(embedding)
    with session_factory() as session:
        img = Image(file_path=f"/test/img_{faiss_id}.jpg", file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            faiss_id=faiss_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=100,
            bbox_h=100,
            detection_confidence=0.9,
            person_id=person_id,
            cluster_id=cluster_id,
            state=FaceState.IDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.commit()
        return img.id, faiss_id


def _add_unidentified_face(
    session_factory,
    vector_store: VectorStore,
    embedding: np.ndarray,
) -> tuple[int, int]:
    """Insert an unidentified Face in FAISS+DB; return (image_id, faiss_id)."""
    faiss_id = vector_store.add(embedding)
    with session_factory() as session:
        img = Image(file_path=f"/test/unid_{faiss_id}.jpg", file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            faiss_id=faiss_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=100,
            bbox_h=100,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.commit()
        return img.id, faiss_id


def test_no_identified_faces_returns_empty(search_db, vs):
    """Person exists with a cluster but no identified faces → empty result."""
    with search_db() as session:
        person = Person(name="Empty Person")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(person_id=person.id)
        session.add(cluster)
        session.commit()
        person_id = person.id

    result = _find_images_by_person(search_db, vs, person_id)
    assert result == []


def test_finds_similar_face(search_db, vs):
    """Person's cluster embedding matches a similar unidentified face."""
    person_id, cluster_id = _add_person_cluster(search_db)

    emb = _rand_norm_emb()
    labelled_image_id, _ = _add_identified_face(
        search_db, vs, person_id, cluster_id, emb
    )
    # Add an unidentified face with the same embedding (should be found)
    similar_image_id, _ = _add_unidentified_face(search_db, vs, emb.copy())

    result = _find_images_by_person(search_db, vs, person_id)
    result_ids = [r[0] for r in result]

    assert labelled_image_id in result_ids
    assert similar_image_id in result_ids
    # Scores should be in descending order
    assert all(result[i][1] >= result[i + 1][1] for i in range(len(result) - 1))


def test_excludes_dissimilar_face(search_db, vs):
    """Face with orthogonal embedding (dot product = 0) is below threshold."""
    person_id, cluster_id = _add_person_cluster(search_db)

    emb_person = np.zeros(512, dtype=np.float32)
    emb_person[0] = 1.0
    _add_identified_face(search_db, vs, person_id, cluster_id, emb_person)

    # Orthogonal embedding → inner product = 0, below any positive threshold
    emb_other = np.zeros(512, dtype=np.float32)
    emb_other[1] = 1.0
    other_image_id, _ = _add_unidentified_face(search_db, vs, emb_other)

    result = _find_images_by_person(search_db, vs, person_id, threshold=0.35)
    result_ids = [r[0] for r in result]

    assert other_image_id not in result_ids


def test_person_with_no_clusters_returns_empty(search_db, vs):
    """Person that has no EmbeddingCluster records → empty result (line 52)."""
    with search_db() as session:
        person = Person(name="Cluster-less")
        session.add(person)
        session.commit()
        person_id = person.id

    result = _find_images_by_person(search_db, vs, person_id)
    assert result == []


def test_stale_faiss_id_causes_indexerror_skipped(search_db, vs):
    """Face with faiss_id absent from VectorStore is skipped (lines 78-79)."""
    person_id, cluster_id = _add_person_cluster(search_db)

    # Insert face with faiss_id=99 but never add that vector to 'vs'
    with search_db() as session:
        img = Image(file_path="/stale_img.jpg", file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            faiss_id=99,  # does not exist in 'vs' (which is empty)
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            person_id=person_id,
            cluster_id=cluster_id,
            state=FaceState.IDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.commit()

    result = _find_images_by_person(search_db, vs, person_id)
    # faiss_id 99 raises IndexError → cluster is skipped → no results
    assert result == []


def test_cluster_with_all_stale_faiss_ids_skipped(search_db, vs):
    """All embeddings raise IndexError → cluster skipped entirely (line 82)."""
    person_id, cluster_id = _add_person_cluster(search_db)

    # Add a valid identified face to a second cluster (to ensure the function
    # progresses past the first cluster) – first cluster has stale faiss_id
    with search_db() as session:
        # Cluster 1 face with invalid faiss_id
        img1 = Image(file_path="/stale1.jpg", file_size=100)
        session.add(img1)
        session.flush()
        face1 = Face(
            image_id=img1.id,
            faiss_id=999,
            bbox_x=0,
            bbox_y=0,
            bbox_w=50,
            bbox_h=50,
            detection_confidence=0.9,
            person_id=person_id,
            cluster_id=cluster_id,
            state=FaceState.IDENTIFIED,
            model_version="test",
        )
        session.add(face1)
        session.commit()

    # vs is empty → faiss_id=999 raises IndexError → embeddings=[] → line 82
    result = _find_images_by_person(search_db, vs, person_id)
    assert result == []


def test_multi_cluster_union(search_db, vs):
    """Person with 2 clusters: images similar to either cluster both appear."""
    with search_db() as session:
        person = Person(name="Multi-Era Person")
        session.add(person)
        session.flush()
        c1 = EmbeddingCluster(person_id=person.id, label="childhood")
        c2 = EmbeddingCluster(person_id=person.id, label="adult")
        session.add_all([c1, c2])
        session.commit()
        person_id = person.id
        cluster1_id = c1.id
        cluster2_id = c2.id

    # Cluster 1: unit vector along axis 0
    emb_child = np.zeros(512, dtype=np.float32)
    emb_child[0] = 1.0
    _add_identified_face(search_db, vs, person_id, cluster1_id, emb_child)

    # Cluster 2: unit vector along axis 1 (orthogonal to cluster 1)
    emb_adult = np.zeros(512, dtype=np.float32)
    emb_adult[1] = 1.0
    _add_identified_face(search_db, vs, person_id, cluster2_id, emb_adult)

    # One unidentified image similar to each cluster
    similar_child_id, _ = _add_unidentified_face(search_db, vs, emb_child.copy())
    similar_adult_id, _ = _add_unidentified_face(search_db, vs, emb_adult.copy())

    result = _find_images_by_person(search_db, vs, person_id)
    result_ids = [r[0] for r in result]

    assert similar_child_id in result_ids
    assert similar_adult_id in result_ids


def test_search_images_gps_only(search_db, tmp_path):
    """search_images filters by GPS when no person_ids are provided."""
    with search_db() as session:
        img1 = Image(file_path="/img1.jpg", file_size=100, file_hash="h1")
        session.add(img1)
        session.flush()
        meta1 = ImageMetadata(
            image_id=img1.id,
            gps_lat=52.52,
            gps_lon=13.40,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta1)

        img2 = Image(file_path="/img2.jpg", file_size=100, file_hash="h2")
        session.add(img2)
        session.flush()
        meta2 = ImageMetadata(
            image_id=img2.id,
            gps_lat=40.0,
            gps_lon=10.0,
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta2)
        session.commit()
        img1_id = img1.id

    bbox = GpsBoundingBox(south=52.0, west=13.0, north=53.0, east=14.0)
    # vs is empty, but we don't pass person_ids
    vs = VectorStore()
    results = search_images(tmp_path, search_db, vs, person_ids=[], gps_bbox=bbox)

    assert len(results) == 1
    assert results[0].image_id == img1_id


def test_search_images_person_only(search_db, vs, tmp_path):
    """search_images filters by person when no GPS is provided."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _rand_norm_emb()
    img_id, _ = _add_identified_face(search_db, vs, person_id, cluster_id, emb)

    results = search_images(
        tmp_path, search_db, vs, person_ids=[person_id], gps_bbox=None
    )

    assert len(results) == 1
    assert results[0].image_id == img_id


def test_search_images_intersection(search_db, vs, tmp_path):
    """search_images intersects person and GPS filters."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _rand_norm_emb()

    # Image 1: has person, inside GPS
    img1_id, _ = _add_identified_face(search_db, vs, person_id, cluster_id, emb)
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
    img2_id, _ = _add_identified_face(search_db, vs, person_id, cluster_id, emb)
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
        tmp_path, search_db, vs, person_ids=[person_id], gps_bbox=bbox
    )

    assert len(results) == 1
    assert results[0].image_id == img1_id


def test_search_images_multiple_persons(search_db, vs, tmp_path):
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
            faiss_id=vs.add(emb1),
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
            faiss_id=vs.add(emb2),
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
    _add_identified_face(search_db, vs, p1_id, c1_id, emb1)

    # Empty person_ids list
    results_empty = search_images(tmp_path, search_db, vs, person_ids=[], gps_bbox=None)
    assert results_empty == []

    # One person (p1)
    results_p1 = search_images(
        tmp_path, search_db, vs, person_ids=[p1_id], gps_bbox=None
    )
    assert len(results_p1) == 2  # both.jpg and img1.jpg

    # Intersection of p1 and p2
    results_both = search_images(
        tmp_path, search_db, vs, person_ids=[p1_id, p2_id], gps_bbox=None
    )
    assert len(results_both) == 1
    assert results_both[0].image_id == both_id

    # Intersection with no common images
    p3_id, c3_id = _add_person_cluster(search_db)
    results_none = search_images(
        tmp_path, search_db, vs, person_ids=[p1_id, p3_id], gps_bbox=None
    )
    assert results_none == []


def test_search_images_ranking(search_db, vs, tmp_path):
    """search_images ranks by minimum score across multiple persons."""
    p1_id, c1_id = _add_person_cluster(search_db)
    p2_id, c2_id = _add_person_cluster(search_db)

    emb_exact = np.zeros(512, dtype=np.float32)
    emb_exact[0] = 1.0

    emb_weak = np.zeros(512, dtype=np.float32)
    emb_weak[0] = 0.5  # Cosine similarity with emb_exact will be 0.5

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
                faiss_id=vs.add(emb_exact),
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
                faiss_id=vs.add(v09),
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
                faiss_id=vs.add(v08),
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
                faiss_id=vs.add(v08),
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
        tmp_path, search_db, vs, person_ids=[p1_id, p2_id], gps_bbox=None
    )
    assert len(results) == 2
    assert results[0].image_id == img1_id
    assert results[1].image_id == img2_id


def test_search_images_empty_input_returns_empty(search_db, vs, tmp_path):
    """search_images returns empty if no filters are provided."""
    results = search_images(tmp_path, search_db, vs, person_ids=[], gps_bbox=None)
    assert results == []
