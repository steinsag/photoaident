"""Tests for person-based image search via FAISS."""

from unittest.mock import patch

import numpy as np

import photoaident.core.search as search_module
from photoaident.core.search import _find_images_by_person, search_images
from photoaident.core.search_person import (
    _intersect_and_rank,
    _match_face_to_person,
    resolve_faces_to_persons,
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
    """Person that has no EmbeddingCluster records → empty result."""
    with search_db() as session:
        person = Person(name="Cluster-less")
        session.add(person)
        session.commit()
        person_id = person.id

    result = _find_images_by_person(search_db, vector_store, person_id)
    assert result == []


def test_stale_faiss_id_causes_indexerror_skipped(search_db, vector_store):
    """Face with faiss_id absent from VectorStore is skipped."""
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
    """All embeddings raise IndexError → cluster skipped entirely."""
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

    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb)

    # Add an unidentified face with the same embedding
    _, unid_faiss_id = _add_unidentified_face(search_db, vector_store, emb.copy())

    results = resolve_faces_to_persons([unid_faiss_id], search_db, vector_store)
    match = results[unid_faiss_id]
    assert match is not None
    assert match[0] == "Test Person"
    assert match[1] > 0.5


def test_resolve_faces_to_persons_no_persons(search_db, vector_store):
    """Returns None for all faces when no persons exist."""
    emb = np.ones(512, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    faiss_id = vector_store.add(emb)

    results = resolve_faces_to_persons([faiss_id], search_db, vector_store)
    assert results[faiss_id] is None


def test_resolve_faces_to_persons_below_threshold(search_db, vector_store):
    """Returns None when similarity is below threshold."""
    person_id, cluster_id = _add_person_cluster(search_db)

    # Person's cluster embedding along axis 0
    emb_person = np.zeros(512, dtype=np.float32)
    emb_person[0] = 1.0
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb_person)

    # Orthogonal embedding → dot product ≈ 0
    emb_other = np.zeros(512, dtype=np.float32)
    emb_other[1] = 1.0
    _, unid_faiss_id = _add_unidentified_face(search_db, vector_store, emb_other)

    results = resolve_faces_to_persons(
        [unid_faiss_id], search_db, vector_store, threshold=0.35
    )
    assert results[unid_faiss_id] is None


def test_resolve_faces_to_persons_empty_input(search_db, vector_store):
    """Returns empty dict for empty input."""
    assert resolve_faces_to_persons([], search_db, vector_store) == {}


def test_resolve_faces_to_persons_index_error(search_db, vector_store):
    """Returns None for faiss_id out of bounds in the vector store."""
    # Need at least one person so we get past the early return
    _add_person_cluster(search_db)

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

    emb_alice = np.zeros(512, dtype=np.float32)
    emb_alice[0] = 1.0
    _add_identified_face(search_db, vector_store, p1_id, c1_id, emb_alice)

    emb_bob = np.zeros(512, dtype=np.float32)
    emb_bob[1] = 1.0
    _add_identified_face(search_db, vector_store, p2_id, c2_id, emb_bob)

    # Unidentified faces matching each person
    _, fid_alice = _add_unidentified_face(search_db, vector_store, emb_alice.copy())
    _, fid_bob = _add_unidentified_face(search_db, vector_store, emb_bob.copy())

    results = resolve_faces_to_persons(
        [fid_alice, fid_bob], search_db, vector_store, threshold=0.0
    )

    match_alice = results[fid_alice]
    assert match_alice is not None
    assert match_alice[0] == "Alice"
    match_bob = results[fid_bob]
    assert match_bob is not None
    assert match_bob[0] == "Bob"


def test_resolve_faces_to_persons_consistent_with_search(search_db, vector_store):
    """A face found by _find_images_by_person should also be resolved by
    resolve_faces_to_persons, ensuring search and detail dialog agree."""
    person_id, cluster_id = _add_person_cluster(search_db)

    emb = _rand_norm_emb()
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb)
    _, unid_faiss_id = _add_unidentified_face(search_db, vector_store, emb.copy())

    # Search finds the image
    search_results = _find_images_by_person(search_db, vector_store, person_id)
    search_image_ids = [r[0] for r in search_results]
    assert len(search_image_ids) > 0

    # Resolution also matches the same face
    resolve_results = resolve_faces_to_persons([unid_faiss_id], search_db, vector_store)
    match = resolve_results[unid_faiss_id]
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
    """Common image IDs are ranked by weakest per-person score."""
    scores_a = {10: 0.9, 20: 0.7, 30: 0.5}
    scores_b = {10: 0.6, 20: 0.8, 30: 0.4}
    result = _intersect_and_rank([scores_a, scores_b])
    # min scores: 10→0.6, 20→0.7, 30→0.4 → sorted desc: [20, 10, 30]
    assert result == [20, 10, 30]


# ===========================================================================
# _match_face_to_person — stale faiss_id
# ===========================================================================


def test_match_face_to_person_index_error(search_db, vector_store):
    """Returns None when faiss_id is not in the vector store."""
    person_id, cluster_id = _add_person_cluster(search_db)

    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    _add_identified_face(search_db, vector_store, person_id, cluster_id, emb)

    # Build person_means the same way resolve_faces_to_persons would
    from photoaident.core.search_person import _load_person_cluster_means

    person_names, person_means = _load_person_cluster_means(search_db, vector_store)
    assert len(person_means) > 0

    # faiss_id 9999 does not exist → IndexError → returns None
    result = _match_face_to_person(9999, person_means, person_names, vector_store, 0.0)
    assert result is None
