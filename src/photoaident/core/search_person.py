"""FAISS-based person search helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select

from photoaident.db.cluster_means import deserialize_embedding
from photoaident.db.database import EmbeddingCluster, Face, Person
from photoaident.db.vector_store import FACE_MATCH_THRESHOLD

_SQLITE_IN_LIMIT = 900  # SQLite bound-parameter limit is 999; stay safely below it

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore


def _compute_cluster_mean(
    cluster_id: int,
    session_factory: "sessionmaker",
) -> "np.ndarray | None":
    """Return persisted mean embedding for a cluster, or None if not yet computed."""
    with session_factory() as session:
        blob = session.scalar(
            select(EmbeddingCluster.mean_embedding).where(
                EmbeddingCluster.id == cluster_id
            )
        )
    if blob is None:
        return None
    return deserialize_embedding(blob)


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
        mean_emb = _compute_cluster_mean(cluster_id, session_factory)
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
    threshold: float = FACE_MATCH_THRESHOLD,
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
        threshold: Minimum cosine similarity to include a result
            (default: ``FACE_MATCH_THRESHOLD``).
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


def _load_person_cluster_means(
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
) -> tuple[dict[int, str], list[tuple[int, "np.ndarray"]]]:
    """Load all persons and their persisted cluster mean embeddings from the DB.

    Returns:
        A tuple of (person_names, person_means) where person_names maps
        person_id → name and person_means is a list of (person_id, mean_embedding).
    """
    with session_factory() as session:
        person_rows = session.execute(select(Person.id, Person.name)).all()
        cluster_rows = session.execute(
            select(
                EmbeddingCluster.person_id,
                EmbeddingCluster.mean_embedding,
            ).where(EmbeddingCluster.mean_embedding.isnot(None))
        ).all()

    person_names: dict[int, str] = {r.id: r.name for r in person_rows}

    person_means: list[tuple[int, np.ndarray]] = []
    for row in cluster_rows:
        if row.person_id not in person_names:
            continue
        person_means.append((row.person_id, deserialize_embedding(row.mean_embedding)))

    return person_names, person_means


def _match_face_to_person(
    fid: int,
    person_means: list[tuple[int, "np.ndarray"]],
    person_names: dict[int, str],
    vector_store: "VectorStore",
    threshold: float,
) -> tuple[str, float] | None:
    """Return (person_name, score) for the best-matching person, or None."""
    try:
        embedding = vector_store.get_embedding(fid)
    except IndexError:
        return None

    best_person_id: int | None = None
    best_score = 0.0

    for person_id, mean_emb in person_means:
        score = float(np.dot(embedding, mean_emb))
        if score >= threshold and score > best_score:
            best_score = score
            best_person_id = person_id

    if best_person_id is not None:
        return (person_names[best_person_id], best_score)
    return None


def resolve_faces_to_persons(
    faiss_ids: list[int],
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
    threshold: float = FACE_MATCH_THRESHOLD,
) -> dict[int, tuple[str, float] | None]:
    """Find the best-matching person for unidentified face embeddings.

    Uses the same cluster-mean approach as the library person search so that
    search results and face highlighting in the detail dialog agree.

    For each face, the dot product (= cosine similarity for L2-normalized
    embeddings) against every person's cluster means is computed.  The
    person whose best cluster mean exceeds *threshold* with the highest
    score wins.

    Args:
        faiss_ids: FAISS index IDs of the faces to resolve.
        session_factory: SQLAlchemy session factory.
        vector_store: FAISS vector store.
        threshold: Minimum similarity score to consider a match
            (default: ``FACE_MATCH_THRESHOLD``).

    Returns:
        A dict mapping each *faiss_id* to ``(person_name, score)`` or ``None``.
    """
    if not faiss_ids:
        return {}

    person_names, person_means = _load_person_cluster_means(
        session_factory, vector_store
    )

    if not person_names or not person_means:
        return dict.fromkeys(faiss_ids)

    return {
        fid: _match_face_to_person(
            fid, person_means, person_names, vector_store, threshold
        )
        for fid in faiss_ids
    }
