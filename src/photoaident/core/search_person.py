"""FAISS-based person search helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select

from photoaident.db.database import EmbeddingCluster, Face, FaceState

_SQLITE_IN_LIMIT = 900  # SQLite bound-parameter limit is 999; stay safely below it

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore


def _compute_cluster_mean(
    cluster_id: int,
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
) -> "np.ndarray | None":
    """Return normalized mean embedding for a cluster, or None if no valid faces."""
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
        return None

    embeddings = []
    for faiss_id in faiss_ids:
        try:
            embeddings.append(vector_store.get_embedding(faiss_id))
        except IndexError:
            continue

    if not embeddings:
        return None

    mean_emb = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb


def _accumulate_faiss_scores(
    cluster_ids: list[int],
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
    limit: int,
    threshold: float,
) -> dict[int, float]:
    """Per cluster: compute mean embedding → FAISS search → keep best score."""
    faiss_scores: dict[int, float] = {}
    for cluster_id in cluster_ids:
        mean_emb = _compute_cluster_mean(cluster_id, session_factory, vector_store)
        if mean_emb is None:
            continue
        for fid, score in vector_store.search(
            mean_emb, k=limit * 3, threshold=threshold
        ):
            if fid not in faiss_scores or score > faiss_scores[fid]:
                faiss_scores[fid] = score
    return faiss_scores


def _faiss_to_image_scores(
    faiss_scores: dict[int, float],
    session_factory: "sessionmaker",
) -> dict[int, float]:
    """Resolve faiss_id → image_id and deduplicate keeping the highest score."""
    all_faiss_ids = list(faiss_scores.keys())
    rows: list = []
    with session_factory() as session:
        for chunk_start in range(0, len(all_faiss_ids), _SQLITE_IN_LIMIT):
            chunk = all_faiss_ids[chunk_start : chunk_start + _SQLITE_IN_LIMIT]
            rows.extend(
                session.execute(
                    select(Face.faiss_id, Face.image_id).where(
                        Face.faiss_id.in_(chunk),
                        Face.deleted_at.is_(None),
                    )
                ).all()
            )

    image_scores: dict[int, float] = {}
    for row_faiss_id, row_image_id in rows:
        score = faiss_scores[row_faiss_id]
        if row_image_id not in image_scores or score > image_scores[row_image_id]:
            image_scores[row_image_id] = score
    return image_scores


def _find_images_by_person(
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

    faiss_scores = _accumulate_faiss_scores(
        cluster_ids, session_factory, vector_store, limit, threshold
    )

    if not faiss_scores:
        return []

    image_scores = _faiss_to_image_scores(faiss_scores, session_factory)

    sorted_pairs = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs[:limit]


def _collect_per_person_scores(
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
    person_ids: list[int],
    filter_image_ids: set[int] | None,
) -> list[dict[int, float]]:
    """Build one score-dict per person, optionally filtered to metadata image IDs."""
    per_person_scores: list[dict[int, float]] = []
    for person_id in person_ids:
        scores: dict[int, float] = {
            img_id: score
            for img_id, score in _find_images_by_person(
                session_factory, vector_store, person_id
            )
            if filter_image_ids is None or img_id in filter_image_ids
        }
        per_person_scores.append(scores)
    return per_person_scores


def _intersect_and_rank(per_person_scores: list[dict[int, float]]) -> list[int]:
    """Return image IDs present in all per-person dicts, ranked by min score desc."""
    if not per_person_scores:
        return []

    common_ids = set(per_person_scores[0].keys())
    for scores in per_person_scores[1:]:
        common_ids &= scores.keys()

    if not common_ids:
        return []

    # Weakest match across persons determines relevance
    ranked = sorted(
        common_ids,
        key=lambda img_id: min(s[img_id] for s in per_person_scores),
        reverse=True,
    )
    return ranked
