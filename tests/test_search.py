"""Tests for core.search.find_images_by_person."""

import numpy as np
import pytest
from sqlalchemy import create_engine

from photoaident.core.search import find_images_by_person
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    Person,
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

    result = find_images_by_person(search_db, vs, person_id)
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

    result = find_images_by_person(search_db, vs, person_id)
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

    result = find_images_by_person(search_db, vs, person_id, threshold=0.35)
    result_ids = [r[0] for r in result]

    assert other_image_id not in result_ids


def test_person_with_no_clusters_returns_empty(search_db, vs):
    """Person that has no EmbeddingCluster records → empty result (line 52)."""
    with search_db() as session:
        person = Person(name="Cluster-less")
        session.add(person)
        session.commit()
        person_id = person.id

    result = find_images_by_person(search_db, vs, person_id)
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

    result = find_images_by_person(search_db, vs, person_id)
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
    result = find_images_by_person(search_db, vs, person_id)
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

    result = find_images_by_person(search_db, vs, person_id)
    result_ids = [r[0] for r in result]

    assert similar_child_id in result_ids
    assert similar_adult_id in result_ids
