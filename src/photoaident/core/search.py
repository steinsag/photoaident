"""Person-based image search using FAISS similarity search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select

from photoaident.db.database import EmbeddingCluster, Face, FaceState

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore


def find_images_by_person(
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
    person_id: int,
    threshold: float = 0.35,
    limit: int = 1000,
) -> list[tuple[int, float]]:
    """Return (image_id, max_similarity_score) pairs sorted by score descending.

    Searches for images likely containing the given person by computing the
    mean embedding of each of their clusters and querying FAISS for similar
    face embeddings.

    Args:
        session_factory: SQLAlchemy session factory.
        vector_store: FAISS vector store holding face embeddings.
        person_id: ID of the person to search for.
        threshold: Minimum cosine similarity to include a result.
        limit: Maximum number of images to return.

    Returns:
        List of (image_id, max_similarity_score) tuples, sorted by score desc.
    """
    # Step 1: Load cluster IDs for the person
    with session_factory() as session:
        cluster_ids = list(
            session.scalars(
                select(EmbeddingCluster.id).where(
                    EmbeddingCluster.person_id == person_id
                )
            ).all()
        )

    if not cluster_ids:
        return []

    # Steps 2–3: Per cluster, collect embeddings → compute mean → search FAISS.
    # Accumulate best score per faiss_id across all clusters.
    faiss_scores: dict[int, float] = {}

    for cluster_id in cluster_ids:
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

        if not faiss_ids:
            continue

        embeddings = []
        for faiss_id in faiss_ids:
            try:
                emb = vector_store.get_embedding(faiss_id)
                embeddings.append(emb)
            except IndexError:
                continue

        if not embeddings:
            continue

        mean_emb = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        for fid, score in vector_store.search(
            mean_emb, k=limit * 3, threshold=threshold
        ):
            if fid not in faiss_scores or score > faiss_scores[fid]:
                faiss_scores[fid] = score

    if not faiss_scores:
        return []

    # Step 5: Bulk-resolve faiss_id → image_id in a single query.
    all_faiss_ids = list(faiss_scores.keys())
    with session_factory() as session:
        rows = session.execute(
            select(Face.faiss_id, Face.image_id).where(
                Face.faiss_id.in_(all_faiss_ids),
                Face.deleted_at.is_(None),
            )
        ).all()

    # Step 6: Deduplicate by image_id, keeping max score.
    image_scores: dict[int, float] = {}
    for row_faiss_id, row_image_id in rows:
        score = faiss_scores[row_faiss_id]
        if row_image_id not in image_scores or score > image_scores[row_image_id]:
            image_scores[row_image_id] = score

    # Step 7: Sort by score descending and return up to limit.
    sorted_pairs = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs[:limit]
