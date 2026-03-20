"""One-time FAISS index migration from positional IDs to database Face IDs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from photoaident.db.database import Face
from photoaident.db.vector_store import VectorStore

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


def rebuild_faiss_with_face_ids(
    old_vector_store: VectorStore,
    session_factory: "sessionmaker",
    dimension: int = VectorStore.DEFAULT_DIMENSION,
) -> VectorStore:
    """Rebuild the FAISS index using Face.id as the FAISS key.

    Reads all non-deleted faces from the database, retrieves their embeddings
    from the old positional index via ``faiss_id``, and inserts them into a new
    ``IndexIDMap2``-backed store keyed by ``Face.id``.

    Faces whose old ``faiss_id`` is missing from the index (orphaned or
    corrupted) are skipped with a debug-level log per face and an info-level
    summary at the end.

    Args:
        old_vector_store: The existing VectorStore with positional IDs.
        session_factory: SQLAlchemy session factory.
        dimension: Embedding dimensionality (default 512).

    Returns:
        A new VectorStore with Face.id-keyed embeddings.
    """
    with session_factory() as session:
        rows = session.execute(
            select(Face.id, Face.faiss_id).where(
                Face.deleted_at.is_(None),
                Face.faiss_id.isnot(None),
            )
        ).all()

    new_store = VectorStore(dimension=dimension)
    migrated = 0
    skipped = 0

    for face_id, faiss_id in rows:
        try:
            embedding = old_vector_store.get_embedding(faiss_id)
            new_store.add(face_id, embedding)
            migrated += 1
        except IndexError:
            skipped += 1
            logger.debug(
                "Skipping face %d: old faiss_id %d not found in index",
                face_id,
                faiss_id,
            )

    logger.info(
        "FAISS migration complete: %d vectors migrated, %d skipped",
        migrated,
        skipped,
    )
    return new_store
