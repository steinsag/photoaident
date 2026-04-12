"""Tests for person-based image search via FAISS."""

from unittest.mock import patch

import pytest

import photoaident.core.search as search_module
from photoaident.core.search import search_images
from photoaident.core.search_person import (
    _intersect_and_rank,
    find_best_person_for_face,
    resolve_faces_to_persons,
    _find_images_by_person,
)
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
    _ones_norm_emb,
    _rand_norm_emb,
    _unit_emb,
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

    emb_person = _unit_emb(0)
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb_person)

    # Orthogonal embedding → inner product = 0, below any positive threshold
    emb_other = _unit_emb(1)
    other_image_id, _ = _add_unidentified_face(search_db, vector_store, emb_other)

    result = _find_images_by_person(search_db, vector_store, person_id, threshold=0.35)
    result_ids = [r[0] for r in result]

    assert other_image_id not in result_ids


def test_person_with_no_clusters_returns_empty(search_db, vector_store):
    """Person that has no EmbeddingCluster records → empty result."""
    with search_db() as session:
        person = Person(name="Cluster-less")
        session.add(person)
        session.commit()
        person_id = person.id

    result = _find_images_by_person(search_db, vector_store, person_id)
    assert result == []


def test_stale_face_id_causes_indexerror_skipped(search_db, vector_store):
    """Face whose id is absent from VectorStore is skipped."""
    person_id, cluster_id = _add_person_cluster(search_db)

    # Insert face but never add its embedding to 'vector_store'
    with search_db() as session:
        img = Image(file_path="/stale_img.jpg", file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
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
    # face.id not in vector_store raises IndexError → cluster is skipped → no results
    assert result == []


def test_cluster_with_all_stale_face_ids_skipped(search_db, vector_store):
    """All embeddings raise IndexError → cluster skipped entirely."""
    person_id, cluster_id = _add_person_cluster(search_db)

    with search_db() as session:
        # Cluster 1 face without embedding in vector store
        img1 = Image(file_path="/stale1.jpg", file_size=100)
        session.add(img1)
        session.flush()
        face1 = Face(
            image_id=img1.id,
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

    # vector_store is empty → face.id raises IndexError → embeddings=[]
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
    emb_child = _unit_emb(0)
    _add_identified_face(search_db, vector_store, person_id, cluster1_id, emb_child)

    # Cluster 2: unit vector along axis 1 (orthogonal to cluster 1)
    emb_adult = _unit_emb(1)
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
        tmp_path,
        search_db,
        vector_store,
        person_ids=[person_id],
        gps_bbox=None,
        date_range=None,
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
            tmp_path,
            search_db,
            vector_store,
            person_ids=[person_id],
            gps_bbox=None,
            date_range=None,
        )

    assert {r.image_id for r in results} == expected_ids


# ===========================================================================
# resolve_faces_to_persons — cluster-mean based face resolution
# ===========================================================================


def test_resolve_faces_to_persons_returns_match(search_db, vector_store):
    """Resolves an unidentified face to a person via cluster mean similarity."""
    person_id, cluster_id = _add_person_cluster(search_db)

    emb = _unit_emb(0)
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb)

    # Add an unidentified face with the same embedding
    _, unid_face_id = _add_unidentified_face(search_db, vector_store, emb.copy())

    results = resolve_faces_to_persons([unid_face_id], search_db, vector_store)
    match = results[unid_face_id]
    assert match is not None
    assert match[0] == "Test Person"
    assert match[1] > 0.5


def test_resolve_faces_to_persons_no_persons(search_db, vector_store):
    """Returns None for all faces when no persons exist."""
    emb = _ones_norm_emb()
    _, face_id = _add_unidentified_face(search_db, vector_store, emb)

    results = resolve_faces_to_persons([face_id], search_db, vector_store)
    assert results[face_id] is None


def test_resolve_faces_to_persons_below_threshold(search_db, vector_store):
    """Returns None when similarity is below threshold."""
    person_id, cluster_id = _add_person_cluster(search_db)

    # Person's cluster embedding along axis 0
    emb_person = _unit_emb(0)
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb_person)

    # Orthogonal embedding → dot product ≈ 0
    emb_other = _unit_emb(1)
    _, unid_face_id = _add_unidentified_face(search_db, vector_store, emb_other)

    results = resolve_faces_to_persons(
        [unid_face_id], search_db, vector_store, threshold=0.35
    )
    assert results[unid_face_id] is None


def test_resolve_faces_to_persons_empty_input(search_db, vector_store):
    """Returns empty dict for empty input."""
    assert resolve_faces_to_persons([], search_db, vector_store) == {}


def test_resolve_faces_to_persons_index_error(search_db, vector_store):
    """Returns None for face_id out of bounds in the vector store."""
    # Add an identified face so person_means is non-empty; otherwise
    # resolve_faces_to_persons returns early before reaching the IndexError path.
    person_id, cluster_id = _add_person_cluster(search_db)
    emb = _unit_emb(0)
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb)

    results = resolve_faces_to_persons([999], search_db, vector_store)
    assert results[999] is None


def test_resolve_faces_to_persons_multiple_faces(search_db, vector_store):
    """Resolves multiple faces in a single call, each to the correct person."""
    # Create two persons with different cluster embeddings
    with search_db() as session:
        p1 = Person(name="Alice")
        p2 = Person(name="Bob")
        session.add_all([p1, p2])
        session.flush()
        c1 = EmbeddingCluster(person_id=p1.id, label="adult")
        c2 = EmbeddingCluster(person_id=p2.id, label="adult")
        session.add_all([c1, c2])
        session.commit()
        p1_id, c1_id = p1.id, c1.id
        p2_id, c2_id = p2.id, c2.id

    emb_alice = _unit_emb(0)
    _add_identified_face(search_db, vector_store, p1_id, c1_id, emb_alice)

    emb_bob = _unit_emb(1)
    _add_identified_face(search_db, vector_store, p2_id, c2_id, emb_bob)

    # Unidentified faces matching each person
    _, face_id_alice = _add_unidentified_face(search_db, vector_store, emb_alice.copy())
    _, face_id_bob = _add_unidentified_face(search_db, vector_store, emb_bob.copy())

    results = resolve_faces_to_persons(
        [face_id_alice, face_id_bob], search_db, vector_store, threshold=0.0
    )

    match_alice = results[face_id_alice]
    assert match_alice is not None
    assert match_alice[0] == "Alice"
    match_bob = results[face_id_bob]
    assert match_bob is not None
    assert match_bob[0] == "Bob"


def test_resolve_faces_to_persons_consistent_with_search(search_db, vector_store):
    """A face found by _find_images_by_person should also be resolved by
    resolve_faces_to_persons, ensuring search and detail dialog agree."""
    person_id, cluster_id = _add_person_cluster(search_db)

    emb = _rand_norm_emb()
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb)
    _, unid_face_id = _add_unidentified_face(search_db, vector_store, emb.copy())

    # Search finds the image
    search_results = _find_images_by_person(search_db, vector_store, person_id)
    search_image_ids = [r[0] for r in search_results]
    assert len(search_image_ids) > 0

    # Resolution also matches the same face
    resolve_results = resolve_faces_to_persons([unid_face_id], search_db, vector_store)
    match = resolve_results[unid_face_id]
    assert match is not None
    assert match[0] == "Test Person"


# ===========================================================================
# _intersect_and_rank — edge cases
# ===========================================================================


def test_intersect_and_rank_empty_list():
    """Empty input returns empty result."""
    assert _intersect_and_rank([]) == []


def test_intersect_and_rank_no_common_ids():
    """Disjoint score dicts return empty result."""
    assert _intersect_and_rank([{1: 0.9}, {2: 0.8}]) == []


def test_intersect_and_rank_multiple_persons():
    """Common image IDs are ranked by weakest per-person score, with scores attached."""
    scores_a = {10: 0.9, 20: 0.7, 30: 0.5}
    scores_b = {10: 0.6, 20: 0.8, 30: 0.4}
    result = _intersect_and_rank([scores_a, scores_b])
    # min scores: 10→0.6, 20→0.7, 30→0.4 → sorted desc: [20, 10, 30]
    ids = [img_id for img_id, _ in result]
    assert ids == [20, 10, 30]
    scores = dict(result)
    assert scores[20] == pytest.approx(0.7)
    assert scores[10] == pytest.approx(0.6)
    assert scores[30] == pytest.approx(0.4)


# ===========================================================================
# resolve_faces_to_persons — stale face_id
# ===========================================================================


def test_resolve_faces_stale_face_id(search_db, vector_store):
    """Returns None for a face_id that is not in the vector store."""
    person_id, cluster_id = _add_person_cluster(search_db)

    emb = _unit_emb(0)
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb)

    # face_id 9999 does not exist in vector store → IndexError path → maps to None
    result = resolve_faces_to_persons([9999], search_db, vector_store, threshold=0.0)
    assert result == {9999: None}


# ===========================================================================
# find_best_person_for_face
# ===========================================================================


def test_find_best_person_for_face_returns_closest(search_db, vector_store):
    """Returns the person_id whose cluster mean is closest to the face embedding."""
    person_a_id, cluster_a_id = _add_person_cluster(search_db)
    _add_identified_face(
        search_db, vector_store, person_a_id, cluster_a_id, _unit_emb(0)
    )

    person_b_id, cluster_b_id = _add_person_cluster(search_db)
    _add_identified_face(
        search_db, vector_store, person_b_id, cluster_b_id, _unit_emb(1)
    )

    _, face_id = _add_unidentified_face(search_db, vector_store, _unit_emb(1))

    result = find_best_person_for_face(face_id, search_db, vector_store)
    assert result == person_b_id


def test_find_best_person_for_face_no_cluster_means(search_db, vector_store):
    """Returns None when no cluster means have been persisted."""
    _, face_id = _add_unidentified_face(search_db, vector_store, _unit_emb(0))
    # No persons/clusters in DB → no means → None
    result = find_best_person_for_face(face_id, search_db, vector_store)
    assert result is None


def test_find_best_person_for_face_missing_embedding(search_db, vector_store):
    """Returns None when the face has no embedding in the vector store."""
    person_id, cluster_id = _add_person_cluster(search_db)
    _add_identified_face(search_db, vector_store, person_id, cluster_id, _unit_emb(0))

    result = find_best_person_for_face(99999, search_db, vector_store)
    assert result is None
