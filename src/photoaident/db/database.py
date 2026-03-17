from datetime import datetime
from enum import Enum as PyEnum
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Numeric,
    UniqueConstraint,
    func,
    create_engine,
    Engine,
    select,
    delete,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    pass


class TakenAtSource(PyEnum):
    EXIF = "exif"
    FILEPATH = "filepath"
    MANUAL = "manual"


class TagSource(PyEnum):
    MODEL = "model"
    MANUAL = "manual"


class FaceState(PyEnum):
    UNIDENTIFIED = "unidentified"
    IDENTIFIED = "identified"
    ANONYMOUS = "anonymous"


class SuggestionState(PyEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


_CASCADE = "all, delete-orphan"
_FK_IMAGES = "images.id"
_FK_PERSONS = "persons.id"

# Ordered list of the 5 canonical age-group slot keys
# (display names are translated in the UI).
AGE_CLUSTERS: list[str] = ["infant", "youngster", "teenager", "adult", "senior"]


class Image(Base):
    """Stores information about indexed image files.

    Relations:
        - metadata_rel: One-to-one with ImageMetadata.
        - tags: One-to-many with ImageTag.
        - faces: One-to-many with Face.
    """

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_path: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    file_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    index_version: Mapped[int] = mapped_column(Integer, default=1)

    metadata_rel: Mapped["ImageMetadata"] = relationship(
        "ImageMetadata",
        back_populates="image",
        uselist=False,
        cascade=_CASCADE,
    )
    tags: Mapped[List["ImageTag"]] = relationship(
        "ImageTag", back_populates="image", cascade=_CASCADE
    )
    faces: Mapped[List["Face"]] = relationship(
        "Face", back_populates="image", cascade=_CASCADE
    )


class ImageMetadata(Base):
    """Stores technical metadata and GPS information for an image.

    Relations:
        - image: Many-to-one with Image (unique).
    """

    __tablename__ = "image_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey(_FK_IMAGES), unique=True, nullable=False
    )
    taken_at: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    taken_at_source: Mapped[Optional[TakenAtSource]] = mapped_column(
        Enum(TakenAtSource, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    camera_make: Mapped[Optional[str]] = mapped_column(String)
    camera_model: Mapped[Optional[str]] = mapped_column(String)
    gps_lat: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=8))
    gps_lon: Mapped[Optional[float]] = mapped_column(Numeric(precision=11, scale=8))
    gps_altitude: Mapped[Optional[float]] = mapped_column(Float)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    orientation: Mapped[int] = mapped_column(Integer, default=1)

    image: Mapped["Image"] = relationship("Image", back_populates="metadata_rel")

    __table_args__ = (
        # Composite index for bounding-box GPS queries (lat/lon range predicates)
        Index("idx_metadata_gps", "gps_lat", "gps_lon"),
    )


class ImageTag(Base):
    """Stores tags and classification results for an image.

    Relations:
        - image: Many-to-one with Image.
    """

    __tablename__ = "image_tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey(_FK_IMAGES), nullable=False)
    tag_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    tag_value: Mapped[str] = mapped_column(
        String, nullable=False
    )  # Stores float confidence as string if needed, or just string
    tag_source: Mapped[TagSource] = mapped_column(
        Enum(TagSource, values_callable=lambda x: [e.value for e in x]), nullable=False
    )
    model_name: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    image: Mapped["Image"] = relationship("Image", back_populates="tags")

    __table_args__ = (
        UniqueConstraint("image_id", "tag_key", "tag_source", name="uq_image_tag"),
    )


class Face(Base):
    """Stores detected face regions and their associations.

    Relations:
        - image: Many-to-one with Image.
        - person: Many-to-one with Person (optional).
        - cluster: Many-to-one with EmbeddingCluster (optional).
        - suggestions: One-to-many with Suggestion.
    """

    __tablename__ = "faces"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey(_FK_IMAGES), nullable=False, index=True
    )
    faiss_id: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    person_id: Mapped[Optional[int]] = mapped_column(ForeignKey(_FK_PERSONS))
    cluster_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("embedding_clusters.id")
    )
    state: Mapped[FaceState] = mapped_column(
        Enum(FaceState, values_callable=lambda x: [e.value for e in x]),
        default=FaceState.UNIDENTIFIED,
        index=True,
    )
    labelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    image: Mapped["Image"] = relationship("Image", back_populates="faces")
    person: Mapped[Optional["Person"]] = relationship("Person", back_populates="faces")
    cluster: Mapped[Optional["EmbeddingCluster"]] = relationship(
        "EmbeddingCluster", back_populates="faces"
    )
    suggestions: Mapped[List["Suggestion"]] = relationship(
        "Suggestion", back_populates="face", cascade=_CASCADE
    )


class Person(Base):
    """Stores information about identified individuals.

    Relations:
        - faces: One-to-many with Face.
        - clusters: One-to-many with EmbeddingCluster.
        - suggestions: One-to-many with Suggestion.
    """

    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    faces: Mapped[List["Face"]] = relationship("Face", back_populates="person")
    clusters: Mapped[List["EmbeddingCluster"]] = relationship(
        "EmbeddingCluster", back_populates="person", cascade=_CASCADE
    )
    suggestions: Mapped[List["Suggestion"]] = relationship(
        "Suggestion", back_populates="person", cascade=_CASCADE
    )


class EmbeddingCluster(Base):
    """Groups face embeddings that likely belong to the same person.

    Relations:
        - person: Many-to-one with Person.
        - faces: One-to-many with Face.
        - suggestions: One-to-many with Suggestion.
    """

    __tablename__ = "embedding_clusters"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey(_FK_PERSONS), nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String)
    age_group: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mean_embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    person: Mapped["Person"] = relationship("Person", back_populates="clusters")
    faces: Mapped[List["Face"]] = relationship("Face", back_populates="cluster")
    suggestions: Mapped[List["Suggestion"]] = relationship(
        "Suggestion", back_populates="cluster", cascade=_CASCADE
    )


class Suggestion(Base):
    """Stores suggested person assignments for faces based on cluster similarity.

    Relations:
        - face: Many-to-one with Face.
        - person: Many-to-one with Person.
        - cluster: Many-to-one with EmbeddingCluster.
    """

    __tablename__ = "suggestions"

    id: Mapped[int] = mapped_column(primary_key=True)
    face_id: Mapped[int] = mapped_column(ForeignKey("faces.id"), nullable=False)
    person_id: Mapped[int] = mapped_column(ForeignKey(_FK_PERSONS), nullable=False)
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("embedding_clusters.id"), nullable=False
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    state: Mapped[SuggestionState] = mapped_column(
        Enum(SuggestionState, values_callable=lambda x: [e.value for e in x]),
        default=SuggestionState.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    face: Mapped["Face"] = relationship("Face", back_populates="suggestions")
    person: Mapped["Person"] = relationship("Person", back_populates="suggestions")
    cluster: Mapped["EmbeddingCluster"] = relationship(
        "EmbeddingCluster", back_populates="suggestions"
    )


def get_engine(db_path: str | None = None) -> Engine:
    """Creates a SQLAlchemy engine for the SQLite database.

    Args:
        db_path: Path to the database file. If None, uses default.
    """
    if db_path is None:
        from photoaident.paths import AppPaths

        db_path = str(AppPaths().db_path)

    return create_engine(f"sqlite:///{db_path}")


def get_session_factory(engine: Engine) -> sessionmaker:
    """Creates a SQLAlchemy session factory.

    Args:
        engine: The SQLAlchemy engine to use.
    """
    return sessionmaker(bind=engine)


def clear_database(session_factory: sessionmaker) -> None:
    """Removes all data from the database.

    Args:
        session_factory: The SQLAlchemy session factory.
    """
    with session_factory() as session:
        # Delete in order to respect foreign keys (if enforced)
        # suggestions -> faces -> images
        # suggestions -> embedding_clusters -> persons
        session.execute(delete(Suggestion))
        session.execute(delete(Face))
        session.execute(delete(ImageTag))
        session.execute(delete(ImageMetadata))
        session.execute(delete(Image))
        session.execute(delete(EmbeddingCluster))
        session.execute(delete(Person))
        session.commit()


def delete_cache_files(
    session_factory: sessionmaker,
    face_crops_dir: Path,
    thumbs_dir: Path,
) -> None:
    """Deletes cached face crop and thumbnail files for all DB entries.

    Only files whose names match a face ID or image hash currently in the
    database are removed. Any other files in those directories are left intact.

    Args:
        session_factory: The SQLAlchemy session factory.
        face_crops_dir: Directory containing face crop JPEG files.
        thumbs_dir: Directory containing photo thumbnail JPEG files.
    """
    with session_factory() as session:
        face_ids = session.execute(select(Face.id)).scalars().all()
        image_hashes = (
            session.execute(select(Image.file_hash).where(Image.file_hash.isnot(None)))
            .scalars()
            .all()
        )

    for face_id in face_ids:
        (face_crops_dir / f"{face_id}.jpg").unlink(missing_ok=True)

    for file_hash in image_hashes:
        (thumbs_dir / f"{file_hash}.jpg").unlink(missing_ok=True)


def get_counts(session_factory: sessionmaker) -> tuple[int, int]:
    """Returns the number of indexed images and discovered faces.

    Args:
        session_factory: The SQLAlchemy session factory.

    Returns:
        A tuple of (image_count, face_count).
    """
    with session_factory() as session:
        image_count = session.scalar(select(func.count(Image.id))) or 0
        face_count = session.scalar(select(func.count(Face.id))) or 0
        return image_count, face_count
