"""Person-based image search using FAISS similarity search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select, and_

from photoaident.core.geo import GpsBoundingBox
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    ImageMetadata,
    Image,
)

if TYPE_CHECKING:
    from pathlib import Path
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore


def search_images(
    thumbs_dir: "Path",
    session_factory: "sessionmaker",
    vector_store: "VectorStore | None",
    person_ids: list[int] | None = None,
    gps_bbox: GpsBoundingBox | None = None,
) -> list[tuple[int, str, "Path"]]:
    """Search for images based on person and/or GPS filters.

    Args:
        thumbs_dir: Directory where thumbnails are stored.
        session_factory: SQLAlchemy session factory.
        vector_store: FAISS vector store. If None, person search is ignored.
        person_ids: List of person IDs to filter by.
        gps_bbox: GPS bounding box to filter by.

    Returns:
        List of (image_id, file_path, thumb_path) tuples.
    """
    gps_image_ids: set[int] | None = None
    if gps_bbox:
        gps_image_ids = set(_find_images_by_gps_bbox(session_factory, gps_bbox))

    if not person_ids and gps_image_ids is None:
        return []

    # Case: GPS only
    if not person_ids and gps_image_ids is not None:
        with session_factory() as session:
            stmt = select(Image).where(Image.id.in_(list(gps_image_ids)))
            images = session.execute(stmt).unique().scalars().all()
            return __format_results(images, thumbs_dir)

    # Case: Person search (possibly intersected with GPS)
    if not vector_store and person_ids:
        # Without vector store, we can't do person search.
        # Original code in library.py fell back to showing all images
        # but only if person_ids was not empty?
        # Actually, LibraryPage tests show it expects all images
        # if VS is None and person is selected.
        with session_factory() as session:
            stmt = select(Image)
            if gps_image_ids is not None:
                stmt = stmt.where(Image.id.in_(list(gps_image_ids)))
            images = session.execute(stmt).unique().scalars().all()
            return __format_results(images, thumbs_dir)

    per_person_scores: list[dict[int, float]] = []
    for person_id in person_ids or []:
        scores: dict[int, float] = {}
        assert vector_store is not None  # Hint for type checker
        for img_id, score in _find_images_by_person(
            session_factory, vector_store, person_id
        ):
            if gps_image_ids is None or img_id in gps_image_ids:
                scores[img_id] = score
        per_person_scores.append(scores)

    if not per_person_scores:
        # This could happen if person_ids was empty but we handled that above.
        # Or if find_images_by_person returned nothing.
        return []

    # Intersection: image must contain ALL selected persons
    common_ids = set(per_person_scores[0].keys())
    for scores in per_person_scores[1:]:
        common_ids &= scores.keys()

    if not common_ids:
        return []

    # Rank by minimum score across persons (weakest match determines relevance)
    image_scores: dict[int, float] = {
        img_id: min(s[img_id] for s in per_person_scores) for img_id in common_ids
    }

    sorted_pairs = sorted(image_scores.items(), key=lambda kv: kv[1], reverse=True)
    ordered_ids = [img_id for img_id, _ in sorted_pairs]

    with session_factory() as session:
        stmt = select(Image).where(Image.id.in_(ordered_ids))
        images = session.execute(stmt).unique().scalars().all()
        image_map = {img.id: img for img in images}
        ordered_images = [image_map[i] for i in ordered_ids if i in image_map]
        return __format_results(ordered_images, thumbs_dir)


def __compute_cluster_mean(
    cluster_id: int,
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
) -> "np.ndarray | None":
    """Return normalised mean embedding for a cluster, or None if no valid faces."""
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


def __accumulate_faiss_scores(
    cluster_ids: list[int],
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
    limit: int,
    threshold: float,
) -> dict[int, float]:
    """Per cluster: compute mean embedding → FAISS search → keep best score."""
    faiss_scores: dict[int, float] = {}
    for cluster_id in cluster_ids:
        mean_emb = __compute_cluster_mean(cluster_id, session_factory, vector_store)
        if mean_emb is None:
            continue
        for fid, score in vector_store.search(
            mean_emb, k=limit * 3, threshold=threshold
        ):
            if fid not in faiss_scores or score > faiss_scores[fid]:
                faiss_scores[fid] = score
    return faiss_scores


def __faiss_to_image_scores(
    faiss_scores: dict[int, float],
    session_factory: "sessionmaker",
) -> dict[int, float]:
    """Resolve faiss_id → image_id and deduplicate keeping the highest score."""
    all_faiss_ids = list(faiss_scores.keys())
    with session_factory() as session:
        rows = session.execute(
            select(Face.faiss_id, Face.image_id).where(
                Face.faiss_id.in_(all_faiss_ids),
                Face.deleted_at.is_(None),
            )
        ).all()

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

    faiss_scores = __accumulate_faiss_scores(
        cluster_ids, session_factory, vector_store, limit, threshold
    )

    if not faiss_scores:
        return []

    image_scores = __faiss_to_image_scores(faiss_scores, session_factory)

    sorted_pairs = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs[:limit]


def _find_images_by_gps_bbox(
    session_factory: "sessionmaker",
    bbox: GpsBoundingBox,
) -> list[int]:
    """Return image IDs within the given GPS bounding box.

    Args:
        session_factory: SQLAlchemy session factory.
        bbox: The GPS bounding box to search in.

    Returns:
        List of image IDs.
    """
    with session_factory() as session:
        if bbox.west <= bbox.east:
            # Simple case
            stmt = select(ImageMetadata.image_id).where(
                and_(
                    ImageMetadata.gps_lat >= bbox.south,
                    ImageMetadata.gps_lat <= bbox.north,
                    ImageMetadata.gps_lon >= bbox.west,
                    ImageMetadata.gps_lon <= bbox.east,
                )
            )
        else:
            # Crosses antimeridian
            stmt = select(ImageMetadata.image_id).where(
                and_(
                    ImageMetadata.gps_lat >= bbox.south,
                    ImageMetadata.gps_lat <= bbox.north,
                    (
                        (ImageMetadata.gps_lon >= bbox.west)
                        | (ImageMetadata.gps_lon <= bbox.east)
                    ),
                )
            )
        return list(session.scalars(stmt).all())


def __format_results(
    images: list[Image], thumbs_dir: "Path"
) -> list[tuple[int, str, "Path"]]:
    """Format Image objects into (id, path, thumb) tuples for the UI."""
    result = []
    for img in images:
        thumb_path = (
            thumbs_dir / f"{img.file_hash}.jpg"
            if img.file_hash
            else thumbs_dir / "unknown.jpg"
        )
        result.append((img.id, img.file_path, thumb_path))
    return result
