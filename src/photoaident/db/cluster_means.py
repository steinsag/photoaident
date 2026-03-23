"""Helpers for persisting and reading cluster mean embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import exists, select

from photoaident.db.database import EmbeddingCluster, Face, FaceState
from photoaident.db.vector_store import VectorStore

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


def serialize_embedding(emb: np.ndarray) -> bytes:
    """Convert a float32 embedding array to raw bytes for DB storage.

    Raises:
        ValueError: if *emb* is not a 1-D array of length
        ``VectorStore.DEFAULT_DIMENSION``.
    """
    if emb.ndim != 1 or emb.shape[0] != VectorStore.DEFAULT_DIMENSION:
        raise ValueError(
            f"serialize_embedding expects shape ({VectorStore.DEFAULT_DIMENSION},),"
            f" got {emb.shape}"
        )
    return emb.astype(VectorStore.EMBEDDING_DTYPE).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray | None:
    """Convert raw bytes back to a float32 embedding array.

    Returns ``None`` and logs a warning if *blob* has an unexpected length,
    so callers can treat the cluster mean as missing rather than crashing at
    runtime. The blob is otherwise assumed to be encoded using
    ``VectorStore.EMBEDDING_DTYPE``.
    """
    expected_bytes = (
        VectorStore.DEFAULT_DIMENSION * np.dtype(VectorStore.EMBEDDING_DTYPE).itemsize
    )
    if len(blob) != expected_bytes:
        logger.warning(
            "deserialize_embedding: expected %d bytes, got %d — treating as missing",
            expected_bytes,
            len(blob),
        )
        return None
    return np.frombuffer(blob, dtype=VectorStore.EMBEDDING_DTYPE).copy()


def recompute_cluster_mean(
    cluster_id: int,
    session_factory: sessionmaker,
    vector_store: VectorStore,
) -> None:
    """Recompute and persist the mean embedding for a single cluster.

    Sets ``EmbeddingCluster.mean_embedding`` to the L2-normalized mean of all
    identified face embeddings in the cluster, or ``NULL`` if the cluster has
    no valid faces.
    """
    with session_factory() as session:
        face_ids = list(
            session.scalars(
                select(Face.id).where(
                    Face.cluster_id == cluster_id,
                    Face.state == FaceState.IDENTIFIED,
                    Face.deleted_at.is_(None),
                )
            ).all()
        )

    mean_blob: bytes | None = None
    if face_ids:
        embeddings: list[np.ndarray] = []
        for face_id in face_ids:
            try:
                embeddings.append(vector_store.get_embedding(face_id))
            except IndexError:
                continue

        if embeddings:
            mean_emb = np.mean(np.stack(embeddings), axis=0).astype(
                VectorStore.EMBEDDING_DTYPE
            )
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-9:
                mean_blob = serialize_embedding(mean_emb / norm)
            else:
                logger.warning(
                    "Cluster %d: mean embedding norm is near-zero (%.2e);"
                    " persisting NULL",
                    cluster_id,
                    norm,
                )

    with session_factory() as session:
        cluster = session.get(EmbeddingCluster, cluster_id)
        if cluster is not None:
            cluster.mean_embedding = mean_blob
            session.commit()


def backfill_cluster_means(
    session_factory: sessionmaker,
    vector_store: VectorStore,
) -> int:
    """Batch-recompute cluster means that are missing.

    Only processes clusters whose ``mean_embedding`` is ``NULL`` and that have
    at least one identified face.  Returns the number of clusters updated.
    """
    with session_factory() as session:
        stmt = select(EmbeddingCluster.id)
        has_identified_face = exists(
            select(Face.id).where(
                Face.cluster_id == EmbeddingCluster.id,
                Face.state == FaceState.IDENTIFIED,
                Face.deleted_at.is_(None),
            )
        )
        stmt = stmt.where(
            EmbeddingCluster.mean_embedding.is_(None),
            has_identified_face,
        )
        cluster_ids = list(session.scalars(stmt).all())

    count = 0
    for cluster_id in cluster_ids:
        recompute_cluster_mean(cluster_id, session_factory, vector_store)
        count += 1

    if count:
        logger.info("Backfilled mean embeddings for %d cluster(s)", count)
    return count
