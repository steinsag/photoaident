"""Tests for photoaident.db.faiss_migration — FAISS positional-to-Face.id migration."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from faiss import IndexFlatIP

from photoaident.db.database import Face, FaceState, Image
from photoaident.db.faiss_migration import rebuild_faiss_with_face_ids
from photoaident.db.vector_store import VectorStore

_DIM = VectorStore.DEFAULT_DIMENSION


def _unit_vector(seed: int = 0) -> np.ndarray:
    """Return a random L2-normalised vector of dimension ``_DIM``."""
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(VectorStore.EMBEDDING_DTYPE)
    v /= np.linalg.norm(v)
    return v


def _old_format_store(embeddings: list[np.ndarray]) -> VectorStore:
    """Build a VectorStore whose inner index is a bare IndexFlatIP (old format).

    Embeddings are added positionally (0, 1, 2, ...) via the raw index,
    simulating the legacy store before the IndexIDMap2 migration.
    """
    store = VectorStore(dimension=_DIM)
    raw = IndexFlatIP(_DIM)
    for emb in embeddings:
        raw.add(emb.reshape(1, -1).astype(VectorStore.EMBEDDING_DTYPE))  # type: ignore[call-arg]
    store.index = raw
    return store


def _insert_image(session) -> Image:
    """Insert a minimal Image row and return it."""
    img = Image(file_path="/fake/img.jpg", file_size=100)
    session.add(img)
    session.flush()
    return img


def _insert_face(
    session,
    image_id: int,
    faiss_id: int | None = None,
    deleted: bool = False,
) -> Face:
    """Insert a minimal Face row and return it."""
    face = Face(
        image_id=image_id,
        faiss_id=faiss_id,
        bbox_x=0,
        bbox_y=0,
        bbox_w=10,
        bbox_h=10,
        detection_confidence=0.95,
        model_version="test-v1",
        state=FaceState.UNIDENTIFIED,
        deleted_at=datetime.now(timezone.utc) if deleted else None,
    )
    session.add(face)
    session.flush()
    return face


# ===========================================================================
# rebuild_faiss_with_face_ids — happy path
# ===========================================================================


def test_migration_remaps_positional_ids_to_face_ids(search_db):
    """Migrate two faces and verify new store uses Face.id as keys."""
    v0 = _unit_vector(0)
    v1 = _unit_vector(1)
    old_store = _old_format_store([v0, v1])

    with search_db() as session:
        img = _insert_image(session)
        f0 = _insert_face(session, img.id, faiss_id=0)
        f1 = _insert_face(session, img.id, faiss_id=1)
        session.commit()
        face_ids = (f0.id, f1.id)

    new_store = rebuild_faiss_with_face_ids(old_store, search_db, dimension=_DIM)

    assert new_store.index.ntotal == 2
    assert np.allclose(new_store.get_embedding(face_ids[0]), v0, atol=1e-5)
    assert np.allclose(new_store.get_embedding(face_ids[1]), v1, atol=1e-5)


def test_migration_returns_indexidmap2(search_db):
    """Migrated store uses IndexIDMap2, not bare IndexFlatIP."""
    old_store = _old_format_store([_unit_vector(0)])

    with search_db() as session:
        img = _insert_image(session)
        _insert_face(session, img.id, faiss_id=0)
        session.commit()

    new_store = rebuild_faiss_with_face_ids(old_store, search_db, dimension=_DIM)

    assert not new_store.needs_migration()


def test_migration_embeddings_are_searchable(search_db):
    """Migrated embeddings can be found via search."""
    v0 = _unit_vector(0)
    old_store = _old_format_store([v0])

    with search_db() as session:
        img = _insert_image(session)
        face = _insert_face(session, img.id, faiss_id=0)
        session.commit()
        face_id = face.id

    new_store = rebuild_faiss_with_face_ids(old_store, search_db, dimension=_DIM)

    results = new_store.search(v0, k=1)
    assert len(results) == 1
    assert results[0][0] == face_id
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# rebuild_faiss_with_face_ids — skip conditions
# ===========================================================================


def test_migration_skips_faces_with_null_faiss_id(search_db):
    """Faces with faiss_id=None are excluded from migration."""
    v0 = _unit_vector(0)
    old_store = _old_format_store([v0])

    with search_db() as session:
        img = _insert_image(session)
        _insert_face(session, img.id, faiss_id=0)
        _insert_face(session, img.id, faiss_id=None)  # should be skipped
        session.commit()

    new_store = rebuild_faiss_with_face_ids(old_store, search_db, dimension=_DIM)

    assert new_store.index.ntotal == 1


def test_migration_skips_deleted_faces(search_db):
    """Soft-deleted faces are excluded from migration."""
    v0 = _unit_vector(0)
    old_store = _old_format_store([v0])

    with search_db() as session:
        img = _insert_image(session)
        _insert_face(session, img.id, faiss_id=0, deleted=True)
        session.commit()

    new_store = rebuild_faiss_with_face_ids(old_store, search_db, dimension=_DIM)

    assert new_store.index.ntotal == 0


def test_migration_skips_out_of_bounds_faiss_id(search_db):
    """Faces whose old faiss_id exceeds index size are skipped gracefully."""
    v0 = _unit_vector(0)
    old_store = _old_format_store([v0])  # only position 0 exists

    with search_db() as session:
        img = _insert_image(session)
        _insert_face(session, img.id, faiss_id=0)
        _insert_face(session, img.id, faiss_id=999)  # out of bounds
        session.commit()

    new_store = rebuild_faiss_with_face_ids(old_store, search_db, dimension=_DIM)

    assert new_store.index.ntotal == 1


# ===========================================================================
# rebuild_faiss_with_face_ids — edge cases
# ===========================================================================


def test_migration_empty_database(search_db):
    """Migration with no face rows produces an empty store."""
    old_store = _old_format_store([])

    new_store = rebuild_faiss_with_face_ids(old_store, search_db, dimension=_DIM)

    assert new_store.index.ntotal == 0


def test_migration_preserves_embedding_values(search_db):
    """Migrated embeddings are numerically identical to originals."""
    vectors = [_unit_vector(i) for i in range(5)]
    old_store = _old_format_store(vectors)

    with search_db() as session:
        img = _insert_image(session)
        faces = [_insert_face(session, img.id, faiss_id=i) for i in range(5)]
        session.commit()
        face_ids = [f.id for f in faces]

    new_store = rebuild_faiss_with_face_ids(old_store, search_db, dimension=_DIM)

    for face_id, original in zip(face_ids, vectors):
        retrieved = new_store.get_embedding(face_id)
        assert np.allclose(retrieved, original, atol=1e-5)


def test_migration_mixed_valid_and_invalid(search_db):
    """Migration handles a mix of valid, null-faiss, deleted, and OOB faces."""
    v0 = _unit_vector(0)
    v1 = _unit_vector(1)
    old_store = _old_format_store([v0, v1])

    with search_db() as session:
        img = _insert_image(session)
        good = _insert_face(session, img.id, faiss_id=0)
        _insert_face(session, img.id, faiss_id=None)  # null
        _insert_face(session, img.id, faiss_id=1, deleted=True)  # deleted
        _insert_face(session, img.id, faiss_id=50)  # OOB
        session.commit()
        good_id = good.id

    new_store = rebuild_faiss_with_face_ids(old_store, search_db, dimension=_DIM)

    assert new_store.index.ntotal == 1
    assert np.allclose(new_store.get_embedding(good_id), v0, atol=1e-5)
