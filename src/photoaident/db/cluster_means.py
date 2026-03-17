"""Helpers for persisting and reading cluster mean embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import exists, select

from photoaident.db.database import EmbeddingCluster, Face, FaceState

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore

logger = logging.getLogger(__name__)

_EMBEDDING_DTYPE = np.float32


def serialize_embedding(emb: np.ndarray) -> bytes:
    """Convert a float32 embedding array to raw bytes for DB storage."""
    return emb.astype(_EMBEDDING_DTYPE).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Convert raw bytes back to a float32 embedding array."""
    return np.frombuffer(blob, dtype=_EMBEDDING_DTYPE).copy()


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
        faiss_ids = list(
            session.scalars(
                select(Face.faiss_id).where(
                    Face.cluster_id == cluster_id,
                    Face.state == FaceState.IDENTIFIED,
                    Face.deleted_at.is_(None),
                )
            ).all()
        )

    mean_blob: bytes | None = None
    if faiss_ids:
        embeddings: list[np.ndarray] = []
        for faiss_id in faiss_ids:
            try:
                embeddings.append(vector_store.get_embedding(faiss_id))
            except IndexError:
                continue

        if embeddings:
            mean_emb = np.mean(np.stack(embeddings), axis=0).astype(_EMBEDDING_DTYPE)
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
    force: bool = False,
) -> int:
    """Batch-recompute cluster means that are missing (or all if *force*).

    Returns the number of clusters updated.
    """
    with session_factory() as session:
        stmt = select(EmbeddingCluster.id)
        if not force:
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
