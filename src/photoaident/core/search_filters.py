"""SQL metadata filter helpers for image search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import and_, func, select

from photoaident.core.date_range import DateRange
from photoaident.core.geo import GpsBoundingBox
from photoaident.db.database import Image, ImageMetadata

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker


def _gps_bbox_subquery(bbox: GpsBoundingBox):
    """Return a SELECT subquery for image_ids within the GPS bounding box.

    Returns an SQLAlchemy select statement (not yet executed) so callers can
    embed it as a subquery without materializing results as bound parameters.
    """
    if bbox.west <= bbox.east:
        return select(ImageMetadata.image_id).where(
            and_(
                ImageMetadata.gps_lat >= bbox.south,
                ImageMetadata.gps_lat <= bbox.north,
                ImageMetadata.gps_lon >= bbox.west,
                ImageMetadata.gps_lon <= bbox.east,
            )
        )
    # Crosses antimeridian
    return select(ImageMetadata.image_id).where(
        and_(
            ImageMetadata.gps_lat >= bbox.south,
            ImageMetadata.gps_lat <= bbox.north,
            (
                (ImageMetadata.gps_lon >= bbox.west)
                | (ImageMetadata.gps_lon <= bbox.east)
            ),
        )
    )


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
        return list(session.scalars(_gps_bbox_subquery(bbox)).all())


def _date_range_subquery(date_range: DateRange):
    """Return a SELECT subquery for image_ids within the given date range.

    Returns an SQLAlchemy select statement (not yet executed) so callers can
    embed it as a subquery without materializing results as bound parameters.
    """
    conditions = []
    start_dt = date_range.to_start_datetime()
    end_dt = date_range.to_end_datetime()

    if start_dt is not None:
        conditions.append(ImageMetadata.taken_at >= start_dt)
    if end_dt is not None:
        conditions.append(ImageMetadata.taken_at <= end_dt)

    # Exclude rows where taken_at is NULL
    conditions.append(ImageMetadata.taken_at.is_not(None))

    return select(ImageMetadata.image_id).where(and_(*conditions))


def _find_images_by_date_range(
    session_factory: "sessionmaker",
    date_range: DateRange,
) -> list[int]:
    """Return image IDs whose taken_at falls within the given date range.

    Args:
        session_factory: SQLAlchemy session factory.
        date_range: The date range to filter by.

    Returns:
        List of image IDs.
    """
    with session_factory() as session:
        return list(session.scalars(_date_range_subquery(date_range)).all())


def _filename_subquery(query: str):
    """Return a SELECT subquery for image_ids matching the filename query.

    Splits the query on whitespace and requires ALL tokens to appear in the
    file_path (case-insensitive AND logic).
    """
    tokens = query.lower().split()
    conditions = [func.lower(Image.file_path).contains(token) for token in tokens]
    return select(Image.id).where(and_(*conditions))


def _find_images_by_filename(
    session_factory: "sessionmaker",
    query: str,
) -> list[int]:
    """Return image IDs whose file_path contains all tokens of the query.

    Splits the query on whitespace; all tokens must match (case-insensitive).

    Args:
        session_factory: SQLAlchemy session factory.
        query: Space-separated substrings to search for in file paths.

    Returns:
        List of image IDs.
    """
    with session_factory() as session:
        return list(session.scalars(_filename_subquery(query)).all())
