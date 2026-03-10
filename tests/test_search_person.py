"""Tests for person-based image search via FAISS."""

from unittest.mock import patch

import numpy as np

import photoaident.core.search as search_module
from photoaident.core.search import _find_images_by_person, search_images
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    Person,
)
from tests.search_helpers import (
    _add_identified_face,
    _add_person_cluster,
    _add_unidentified_face,
    _rand_norm_emb,
)


def test_no_identified_faces_returns_empty(search_db, vector_store):
    """Person exists with a cluster but no identified faces → empty result."""
    with search_db() as session:
        person = Person(name="Empty Person")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(person_id=person.id)
        session.add(cluster)
        session.commit()
        person_id = person.id

    result = _find_images_by_person(search_db, vector_store, person_id)
    assert result == []


def test_finds_similar_face(search_db, vector_store):
    """Person's cluster embedding matches a similar unidentified face."""
    person_id, cluster_id = _add_person_cluster(search_db)

    emb = _rand_norm_emb()
    labelled_image_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )
    # Add an unidentified face with the same embedding (should be found)
    similar_image_id, _ = _add_unidentified_face(search_db, vector_store, emb.copy())

    result = _find_images_by_person(search_db, vector_store, person_id)
    result_ids = [r[0] for r in result]

    assert labelled_image_id in result_ids
    assert similar_image_id in result_ids
    # Scores should be in descending order
    assert all(result[i][1] >= result[i + 1][1] for i in range(len(result) - 1))


def test_excludes_dissimilar_face(search_db, vector_store):
    """Face with orthogonal embedding (dot product = 0) is below threshold."""
    person_id, cluster_id = _add_person_cluster(search_db)

    emb_person = np.zeros(512, dtype=np.float32)
    emb_person[0] = 1.0
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb_person)

    # Orthogonal embedding → inner product = 0, below any positive threshold
    emb_other = np.zeros(512, dtype=np.float32)
    emb_other[1] = 1.0
    other_image_id, _ = _add_unidentified_face(search_db, vector_store, emb_other)

    result = _find_images_by_person(search_db, vector_store, person_id, threshold=0.35)
    result_ids = [r[0] for r in result]

    assert other_image_id not in result_ids


def test_person_with_no_clusters_returns_empty(search_db, vector_store):
    """Person that has no EmbeddingCluster records → empty result (line 52)."""
    with search_db() as session:
        person = Person(name="Cluster-less")
        session.add(person)
        session.commit()
        person_id = person.id

    result = _find_images_by_person(search_db, vector_store, person_id)
    assert result == []


def test_stale_faiss_id_causes_indexerror_skipped(search_db, vector_store):
    """Face with faiss_id absent from VectorStore is skipped (lines 78-79)."""
    person_id, cluster_id = _add_person_cluster(search_db)

    # Insert face with faiss_id=99 but never add that vector to 'vector_store'
    with search_db() as session:
        img = Image(file_path="/stale_img.jpg", file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            faiss_id=99,  # does not exist in 'vector_store' (which is empty)
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

    result = _find_images_by_person(search_db, vector_store, person_id)
    # faiss_id 99 raises IndexError → cluster is skipped → no results
    assert result == []


def test_cluster_with_all_stale_faiss_ids_skipped(search_db, vector_store):
    """All embeddings raise IndexError → cluster skipped entirely (line 82)."""
    person_id, cluster_id = _add_person_cluster(search_db)

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

    # vector_store is empty → faiss_id=999 raises IndexError → embeddings=[] → line 82
    result = _find_images_by_person(search_db, vector_store, person_id)
    assert result == []


def test_multi_cluster_union(search_db, vector_store):
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
    _add_identified_face(search_db, vector_store, person_id, cluster1_id, emb_child)

    # Cluster 2: unit vector along axis 1 (orthogonal to cluster 1)
    emb_adult = np.zeros(512, dtype=np.float32)
    emb_adult[1] = 1.0
    _add_identified_face(search_db, vector_store, person_id, cluster2_id, emb_adult)

    # One unidentified image similar to each cluster
    similar_child_id, _ = _add_unidentified_face(
        search_db, vector_store, emb_child.copy()
    )
    similar_adult_id, _ = _add_unidentified_face(
        search_db, vector_store, emb_adult.copy()
    )

    result = _find_images_by_person(search_db, vector_store, person_id)
    result_ids = [r[0] for r in result]

    assert similar_child_id in result_ids
    assert similar_adult_id in result_ids


def test_search_images_person_only(search_db, vector_store, tmp_path):
    """search_images filters by person when no GPS is provided."""
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _rand_norm_emb()
    img_id, _ = _add_identified_face(
        search_db, vector_store, person_id, cluster_id, emb
    )

    results = search_images(
        tmp_path, search_db, vector_store, person_ids=[person_id], gps_bbox=None
    )

    assert len(results) == 1
    assert results[0].image_id == img_id


def test_search_images_chunks_large_id_list(search_db, vector_store, tmp_path):
    """search_images chunks ordered_ids to stay within SQLite's parameter limit.

    Patches _SQLITE_IN_LIMIT to 3 and creates 5 matching images, verifying that
    all 5 results are returned correctly despite requiring multiple DB round-trips.
    """
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _rand_norm_emb()

    expected_ids = set()
    for _ in range(5):
        img_id, _ = _add_identified_face(
            search_db, vector_store, person_id, cluster_id, emb
        )
        expected_ids.add(img_id)

    with patch.object(search_module, "_SQLITE_IN_LIMIT", 3):
        results = search_images(
            tmp_path, search_db, vector_store, person_ids=[person_id], gps_bbox=None
        )

    assert {r.image_id for r in results} == expected_ids
