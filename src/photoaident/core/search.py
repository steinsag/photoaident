"""Image search orchestration: person, GPS, date, and filename filters."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import and_, select
from sqlalchemy.orm import joinedload

from photoaident.core.date_range import DateRange
from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search_filters import (
    _date_range_subquery,
    _filename_filter_clauses,
    _find_images_by_date_range,
    _find_images_by_filename,
    _find_images_by_gps_bbox,
    _gps_bbox_subquery,
)
from photoaident.core.search_person import (
    _SQLITE_IN_LIMIT,
    _collect_per_person_scores,
    _intersect_and_rank,
)
from photoaident.db.database import Image

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore


class SortOrder(enum.Enum):
    """Sort order options for search results displayed in ThumbnailGrid."""

    RELEVANCE_DESC = "relevance_desc"
    RELEVANCE_ASC = "relevance_asc"
    TAKEN_AT_DESC = "taken_at_desc"
    TAKEN_AT_ASC = "taken_at_asc"
    FILENAME_DESC = "filename_desc"
    FILENAME_ASC = "filename_asc"


@dataclass
class SearchResult:
    """A single search result referencing an image."""

    image_id: int
    file_path: str
    thumb_path: Path
    score: float | None = None
    taken_at: datetime | None = None


def sort_results(results: list[SearchResult], order: SortOrder) -> list[SearchResult]:
    """Return a new sorted list of SearchResult objects for the given order.

    None scores are always placed last regardless of sort direction.
    None taken_at values are always placed last regardless of sort direction.

    Args:
        results: The results to sort.
        order: The desired sort order.

    Returns:
        A new sorted list (the original list is not modified).
    """
    if order == SortOrder.RELEVANCE_DESC:
        return sorted(
            results,
            key=lambda r: (r.score is not None, r.score or 0.0),
            reverse=True,
        )
    if order == SortOrder.RELEVANCE_ASC:
        return sorted(
            results,
            key=lambda r: (r.score is None, r.score or 0.0),
        )
    if order == SortOrder.TAKEN_AT_DESC:
        return sorted(
            results,
            key=lambda r: (r.taken_at is not None, r.taken_at or datetime.min),
            reverse=True,
        )
    if order == SortOrder.TAKEN_AT_ASC:
        return sorted(
            results,
            key=lambda r: (r.taken_at is None, r.taken_at or datetime.min),
        )
    if order == SortOrder.FILENAME_DESC:
        return sorted(results, key=lambda r: r.file_path.lower(), reverse=True)
    # FILENAME_ASC
    return sorted(results, key=lambda r: r.file_path.lower())


def search_images(
    thumbs_dir: Path,
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
    person_ids: list[int],
    gps_bbox: GpsBoundingBox | None,
    date_range: DateRange | None,
    filename_query: str | None = None,
) -> list[SearchResult]:
    """Search for images based on person, GPS, date, and/or filename filters.

    Args:
        thumbs_dir: Directory where thumbnails are stored.
        session_factory: SQLAlchemy session factory.
        vector_store: FAISS vector store.
        person_ids: List of person IDs to filter by.
        gps_bbox: GPS bounding box to filter by.
        date_range: Date range to filter by.
        filename_query: Optional, case-insensitive filename filter. The string is
            split on whitespace into tokens, and each token must appear as a
            substring in the image file path (logical AND).

    Returns:
        List of SearchResult objects.
    """
    filename_query = (
        filename_query.strip() or None if filename_query is not None else None
    )

    if not person_ids and not gps_bbox and not date_range and not filename_query:
        return []

    if not person_ids:
        # Pure metadata query — use SQL subqueries; no Python sets needed.
        return _search_by_metadata_only(
            session_factory, gps_bbox, date_range, filename_query, thumbs_dir
        )

    # Person-based search: materialize metadata IDs so FAISS results can be
    # filtered in Python (FAISS lives outside SQL, so this is unavoidable).
    metadata_ids: set[int] | None = None
    if gps_bbox is not None:
        metadata_ids = set(_find_images_by_gps_bbox(session_factory, gps_bbox))
    if date_range is not None:
        date_ids = set(_find_images_by_date_range(session_factory, date_range))
        metadata_ids = date_ids if metadata_ids is None else metadata_ids & date_ids
    if filename_query is not None:
        filename_ids = set(_find_images_by_filename(session_factory, filename_query))
        metadata_ids = (
            filename_ids if metadata_ids is None else metadata_ids & filename_ids
        )

    per_person_scores = _collect_per_person_scores(
        session_factory, vector_store, person_ids, metadata_ids
    )
    ordered_pairs = _intersect_and_rank(per_person_scores)
    if not ordered_pairs:
        return []

    ordered_ids = [img_id for img_id, _ in ordered_pairs]
    scores = dict(ordered_pairs)
    images = _fetch_ordered_images(session_factory, ordered_ids)
    return _format_results(images, thumbs_dir, scores)


def _search_by_metadata_only(
    session_factory: "sessionmaker",
    gps_bbox: GpsBoundingBox | None,
    date_range: DateRange | None,
    filename_query: str | None,
    thumbs_dir: Path,
) -> list[SearchResult]:
    """Return results for a metadata-only query using SQL subqueries.

    Builds an IN-subquery for each active filter so no large bound-parameter
    sets are passed to SQLite and no Python-side set materialisation is needed.
    """
    conditions = []
    if gps_bbox is not None:
        conditions.append(Image.id.in_(_gps_bbox_subquery(gps_bbox)))
    if date_range is not None:
        conditions.append(Image.id.in_(_date_range_subquery(date_range)))
    if filename_query is not None:
        # Apply clauses directly on the images table — avoids a redundant
        # self-subquery (images.id IN (SELECT images.id FROM images WHERE …)).
        conditions.extend(_filename_filter_clauses(filename_query))
    if not conditions:
        return []
    with session_factory() as session:
        stmt = (
            select(Image)
            .options(joinedload(Image.metadata_rel))
            .where(and_(*conditions))
            .order_by(Image.id)
        )
        images: list[Image] = list(session.execute(stmt).unique().scalars().all())
    return _format_results(images, thumbs_dir)


def _fetch_ordered_images(
    session_factory: "sessionmaker",
    ordered_ids: list[int],
) -> list[Image]:
    """Fetch Image rows for the given IDs, preserving the requested order."""
    with session_factory() as session:
        image_map: dict[int, Image] = {}
        for chunk_start in range(0, len(ordered_ids), _SQLITE_IN_LIMIT):
            chunk = ordered_ids[chunk_start : chunk_start + _SQLITE_IN_LIMIT]
            for img in (
                session.execute(
                    select(Image)
                    .options(joinedload(Image.metadata_rel))
                    .where(Image.id.in_(chunk))
                )
                .unique()
                .scalars()
                .all()
            ):
                image_map[img.id] = img
    return [image_map[i] for i in ordered_ids if i in image_map]


def _format_results(
    images: list[Image],
    thumbs_dir: Path,
    scores: dict[int, float] | None = None,
) -> list[SearchResult]:
    """Format Image objects into SearchResult objects for the UI."""
    return [
        SearchResult(
            image_id=img.id,
            file_path=img.file_path,
            thumb_path=(
                thumbs_dir / f"{img.file_hash}.jpg"
                if img.file_hash
                else thumbs_dir / "unknown.jpg"
            ),
            score=scores[img.id] if scores is not None else None,
            taken_at=img.metadata_rel.taken_at if img.metadata_rel else None,
        )
        for img in images
    ]
