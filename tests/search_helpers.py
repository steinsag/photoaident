"""Shared helper functions for search-layer tests.

Fixtures (search_db, vector_store) are in conftest.py so pytest auto-discovers them
without ruff flagging the imports as unused.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from photoaident.db.cluster_means import recompute_cluster_mean
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    ImageMetadata,
    Person,
    TakenAtSource,
)
from photoaident.db.vector_store import VectorStore

_rng = np.random.default_rng(seed=42)

_DIM = VectorStore.DEFAULT_DIMENSION
_DTYPE = VectorStore.EMBEDDING_DTYPE

_path_counter = 0


def _next_path(prefix: str) -> str:
    """Return a unique file path for test images."""
    global _path_counter
    _path_counter += 1
    return f"/test/{prefix}_{_path_counter}.jpg"


def _rand_norm_emb() -> np.ndarray:
    """Return a random unit-length embedding vector."""
    v = _rng.random(_DIM).astype(_DTYPE)
    v /= np.linalg.norm(v)
    return v


def _zero_emb() -> np.ndarray:
    """Return a zero embedding vector."""
    return np.zeros(_DIM, dtype=_DTYPE)


def _unit_emb(axis: int) -> np.ndarray:
    """Return a unit vector with 1.0 at the given axis index."""
    v = np.zeros(_DIM, dtype=_DTYPE)
    v[axis] = 1.0
    return v


def _ones_norm_emb() -> np.ndarray:
    """Return a normalized all-ones embedding vector."""
    v = np.ones(_DIM, dtype=_DTYPE)
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
    """Insert an identified Face and update the cluster mean.

    Returns (image_id, face_id).
    """
    with session_factory() as session:
        img = Image(file_path=_next_path("img"), file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
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
        session.flush()
        face_id = face.id
        image_id = img.id
        vector_store.add(face_id, embedding)
        session.commit()
    recompute_cluster_mean(cluster_id, session_factory, vector_store)
    return image_id, face_id


def _add_unidentified_face(
    session_factory,
    vector_store: VectorStore,
    embedding: np.ndarray,
) -> tuple[int, int]:
    """Insert an unidentified Face in FAISS+DB; return (image_id, face_id)."""
    with session_factory() as session:
        img = Image(file_path=_next_path("unid"), file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=100,
            bbox_h=100,
            detection_confidence=0.9,
            state=FaceState.UNIDENTIFIED,
            model_version="test",
        )
        session.add(face)
        session.flush()
        face_id = face.id
        image_id = img.id
        vector_store.add(face_id, embedding)
        session.commit()
        return image_id, face_id


def _add_face_for_image(
    session_factory,
    vector_store: VectorStore,
    person_id: int,
    cluster_id: int,
    file_path: str,
    embedding: np.ndarray,
) -> int:
    """Insert an Image with the given path and an identified Face; return image_id."""
    from photoaident.db.database import Face, FaceState, Image

    with session_factory() as session:
        img = Image(file_path=file_path, file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
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
        session.flush()
        vector_store.add(face.id, embedding)
        session.commit()
        image_id = img.id
    recompute_cluster_mean(cluster_id, session_factory, vector_store)
    return image_id


def _add_image_with_metadata(
    session_factory,
    file_path: str = "/img.jpg",
    file_hash: str = "hash",
    taken_at: datetime | None = None,
    gps_lat: float | None = None,
    gps_lon: float | None = None,
) -> int:
    """Insert an Image with optional ImageMetadata; return image_id."""
    with session_factory() as session:
        img = Image(file_path=file_path, file_size=100, file_hash=file_hash)
        session.add(img)
        session.flush()
        meta = ImageMetadata(
            image_id=img.id,
            taken_at=taken_at,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            taken_at_source=TakenAtSource.EXIF,
            width=100,
            height=100,
        )
        session.add(meta)
        session.commit()
        return img.id
