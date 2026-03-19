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
    face_scores: dict[int, float],
    session_factory: "sessionmaker",
) -> dict[int, float]:
    """Resolve face_id → image_id and deduplicate keeping the highest score."""
    all_face_ids = list(face_scores.keys())
    rows: list = []
    with session_factory() as session:
        for chunk_start in range(0, len(all_face_ids), _SQLITE_IN_LIMIT):
            chunk = all_face_ids[chunk_start : chunk_start + _SQLITE_IN_LIMIT]
            rows.extend(
                session.execute(
                    select(Face.id, Face.image_id).where(
                        Face.id.in_(chunk),
                        Face.deleted_at.is_(None),
                    )
                ).all()
            )

    image_scores: dict[int, float] = {}
    for row_face_id, row_image_id in rows:
        score = face_scores[row_face_id]
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
) -> tuple[dict[int, str], list[tuple[int, "np.ndarray"]]]:
    """Load persons and their persisted cluster mean embeddings from the DB.

    Issues a single joined query filtered to clusters with non-NULL means,
    returning (person_id, person_name, mean_embedding) rows directly.

    Returns:
        A tuple of (person_names, person_means) where person_names maps
        person_id → name and person_means is a list of (person_id, mean_embedding).
    """
    with session_factory() as session:
        rows = session.execute(
            select(Person.id, Person.name, EmbeddingCluster.mean_embedding)
            .join(EmbeddingCluster, EmbeddingCluster.person_id == Person.id)
            .where(EmbeddingCluster.mean_embedding.isnot(None))
        ).all()

    person_names: dict[int, str] = {}
    person_means: list[tuple[int, np.ndarray]] = []
    for person_id, person_name, blob in rows:
        mean_vec = deserialize_embedding(blob)
        if mean_vec is None:
            continue
        person_names[person_id] = person_name
        person_means.append((person_id, mean_vec))

    return person_names, person_means


def resolve_faces_to_persons(
    face_ids: list[int],
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
    threshold: float = FACE_MATCH_THRESHOLD,
) -> dict[int, tuple[str, float] | None]:
    """Find the best-matching person for unidentified face embeddings.

    Uses the same cluster-mean approach as the library person search so that
    search results and face highlighting in the detail dialog agree.

    Embeddings are batched into an (N, D) matrix (where D is the embedding
    dimension) and scored against all M cluster means in a single
    (N, D) @ (M, D).T multiply, reducing Python overhead to O(1) NumPy/BLAS
    calls instead of O(N × M) Python iterations.

    Args:
        face_ids: Database Face.id values of the faces to resolve.
        session_factory: SQLAlchemy session factory.
        vector_store: FAISS vector store.
        threshold: Minimum similarity score to consider a match
            (default: ``FACE_MATCH_THRESHOLD``).

    Returns:
        A dict mapping each *face_id* to ``(person_name, score)`` or ``None``.
    """
    if not face_ids:
        return {}

    person_names, person_means = _load_person_cluster_means(session_factory)

    if not person_names or not person_means:
        return dict.fromkeys(face_ids)

    # Validate that the vector store's embedding dimension matches the
    # persisted cluster means' dimension so that emb_matrix @ means_matrix.T
    # is well-defined.
    means_dim = person_means[0][1].shape[0]
    if hasattr(vector_store, "dimension") and vector_store.dimension != means_dim:
        raise ValueError(
            f"VectorStore dimension ({vector_store.dimension}) does not match "
            f"persisted person cluster means dimension ({means_dim}). "
            "Ensure both use the same embedding size."
        )

    # Build (M, D) means matrix once; keep parallel person_id list for lookup.
    mean_person_ids = [pid for pid, _ in person_means]
    means_matrix = np.stack([emb for _, emb in person_means])  # (M, D)

    result: dict[int, tuple[str, float] | None] = {}
    valid_fids: list[int] = []
    embeddings: list[np.ndarray] = []

    for fid in face_ids:
        try:
            embeddings.append(vector_store.get_embedding(fid))
            valid_fids.append(fid)
        except IndexError:
            result[fid] = None

    if valid_fids:
        emb_matrix = np.stack(embeddings)  # (N, D)
        scores = emb_matrix @ means_matrix.T  # (N, M)

        best_cols = np.argmax(scores, axis=1)  # (N,)
        best_scores = scores[np.arange(len(valid_fids)), best_cols]  # (N,)

        for i, fid in enumerate(valid_fids):
            score = float(best_scores[i])
            if score >= threshold:
                person_id = mean_person_ids[int(best_cols[i])]
                result[fid] = (person_names[person_id], score)
            else:
                result[fid] = None

    return result
