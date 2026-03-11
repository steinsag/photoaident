"""Shared helper functions for search-layer tests.

Fixtures (search_db, vector_store) are in conftest.py so pytest auto-discovers them
without ruff flagging the imports as unused.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

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
            taken_at_source=TakenAtSource.FILESYSTEM,
            width=100,
            height=100,
        )
        session.add(meta)
        session.commit()
        return img.id
