import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import exifread
import numpy as np
from PySide6 import QtCore
from sqlalchemy import select, func, update, delete
from sqlalchemy.orm import sessionmaker

from photoaident.core.embeddings import FaceEmbedder
from photoaident.core.filepath_date import compile_pattern, extract_date_from_path
from photoaident.db.database import Image, Face, FaceState, ImageMetadata, TakenAtSource
from photoaident.db.vector_store import VectorStore
from photoaident.paths import AppPaths
from photoaident.utils.image_utils import open_image

logger = logging.getLogger(__name__)

_EXIF_DATE_TAGS = ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime")
_EXIF_DATE_FORMAT = "%Y:%m:%d %H:%M:%S"


def _dms_to_decimal(values, ref: str) -> float | None:
    """Convert DMS (degrees/minutes/seconds) GPS tag values to decimal degrees."""
    try:
        d = float(values[0].num) / float(values[0].den)
        m = float(values[1].num) / float(values[1].den)
        s = float(values[2].num) / float(values[2].den)
        result = d + m / 60 + s / 3600
        if ref in ("S", "W"):
            result = -result
        return result
    except Exception:
        return None


def _extract_gps_info(tags: dict) -> tuple[float | None, float | None, float | None]:
    """Extract GPS latitude, longitude, and altitude from EXIF tags."""
    lat_tag = tags.get("GPS GPSLatitude")
    lon_tag = tags.get("GPS GPSLongitude")
    gps_lat = gps_lon = None
    if lat_tag and lon_tag:
        lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
        lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))
        gps_lat = _dms_to_decimal(lat_tag.values, lat_ref)
        gps_lon = _dms_to_decimal(lon_tag.values, lon_ref)

    gps_altitude = None
    alt_tag = tags.get("GPS GPSAltitude")
    if alt_tag:
        try:
            alt_val = float(alt_tag.values[0].num) / float(alt_tag.values[0].den)
            alt_ref_tag = tags.get("GPS GPSAltitudeRef")
            if alt_ref_tag and str(alt_ref_tag) == "1":
                alt_val = -alt_val
            gps_altitude = alt_val
        except Exception:
            pass

    return gps_lat, gps_lon, gps_altitude


def _extract_camera_info(tags: dict) -> tuple[str | None, str | None, int]:
    """Extract camera make, model, and image orientation from EXIF tags."""
    camera_make = str(tags["Image Make"]) if "Image Make" in tags else None
    camera_model = str(tags["Image Model"]) if "Image Model" in tags else None

    orientation = 1
    if "Image Orientation" in tags:
        try:
            orientation = int(tags["Image Orientation"].values[0])
        except (ValueError, TypeError, AttributeError):
            pass
    return camera_make, camera_model, orientation


class IndexingTask(QtCore.QObject):
    """Background task that detects faces in images, computes embeddings,
    and updates the database and FAISS index."""

    progress = QtCore.Signal(int, int, int, str)  # indexed, total, faces, status
    finished = QtCore.Signal()

    def __init__(
        self,
        session_factory: sessionmaker,
        vector_store: VectorStore,
        paths: AppPaths,
        ctx_id: int = 0,
        filepath_date_pattern: str = "",
    ):
        super().__init__()
        self.session_factory = session_factory
        self.vector_store = vector_store
        self.paths = paths
        self.ctx_id = ctx_id
        self._is_cancelled = False
        self._embedder: FaceEmbedder | None = None
        self._compiled_pattern: re.Pattern[str] | None = None
        if filepath_date_pattern:
            try:
                self._compiled_pattern = compile_pattern(filepath_date_pattern)
            except ValueError:
                logger.warning(
                    "Invalid filepath_date_pattern %r; path date extraction disabled",
                    filepath_date_pattern,
                )

    def cancel(self) -> None:
        """Signal the task to stop after the current image finishes."""
        self._is_cancelled = True

    def _get_embedder(self) -> FaceEmbedder:
        """Return the shared embedder, initializing it lazily on first use."""
        if self._embedder is None:
            self._embedder = FaceEmbedder(ctx_id=self.ctx_id)
        return self._embedder

    def _calculate_hash(self, file_path: Path) -> str:
        """Compute and return the SHA-256 hex digest of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_taken_at(
        self, tags: dict, path: Path
    ) -> tuple[datetime | None, TakenAtSource | None]:
        """Determine when the image was taken.

        Fallback chain:
        1. EXIF tags (DateTimeOriginal / DateTimeDigitized / DateTime)
        2. Filepath pattern match (only when a pattern is configured)
        3. ``(None, None)`` — no reliable date available
        """
        for tag_key in _EXIF_DATE_TAGS:
            if tag_key in tags:
                try:
                    taken_at = datetime.strptime(str(tags[tag_key]), _EXIF_DATE_FORMAT)
                    return taken_at, TakenAtSource.EXIF
                except ValueError:
                    continue

        if self._compiled_pattern is not None:
            extracted = extract_date_from_path(path, self._compiled_pattern)
            if extracted is not None:
                logger.debug("Filepath date extracted from %s: %s", path, extracted)
                return (
                    datetime(extracted.year, extracted.month, extracted.day),
                    TakenAtSource.FILEPATH,
                )
            logger.debug("Filepath date pattern did not match: %s", path)
        else:
            logger.debug("No filepath date pattern configured; skipping for %s", path)

        return None, None

    def _extract_exif_metadata(self, path: Path, image_id: int, session) -> None:
        """Extract EXIF metadata from an image file and persist it to image_metadata."""
        try:
            with open_image(path) as pil_img:
                width, height = pil_img.size

            with open(path, "rb") as f:
                tags = exifread.process_file(f, details=False)

            taken_at, taken_at_source = self._get_taken_at(tags, path)
            gps_lat, gps_lon, gps_altitude = _extract_gps_info(tags)
            camera_make, camera_model, orientation = _extract_camera_info(tags)

            session.execute(
                delete(ImageMetadata).where(ImageMetadata.image_id == image_id)
            )
            session.add(
                ImageMetadata(
                    image_id=image_id,
                    taken_at=taken_at,
                    taken_at_source=taken_at_source,
                    camera_make=camera_make,
                    camera_model=camera_model,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                    gps_altitude=gps_altitude,
                    width=width,
                    height=height,
                    orientation=orientation,
                )
            )
        except Exception:
            logger.warning("EXIF extraction failed for %s", path, exc_info=True)

    def _save_face_crops(
        self,
        faces: list[tuple[Face, list]],
        embedder: FaceEmbedder,
        image_path: Path,
    ) -> None:
        """Save JPEG crops for each newly detected face."""
        self.paths.face_crops_dir.mkdir(parents=True, exist_ok=True)
        for face, bbox in faces:
            crop = embedder.extract_face_crop(image_path, bbox)
            crop_path = self.paths.face_crops_dir / f"{face.id}.jpg"
            crop.save(crop_path, "JPEG", quality=90)

    def _fetch_existing_face_data(
        self, session, image_id: int
    ) -> tuple[list[int], dict[int, np.ndarray]]:
        """Return (face_ids, embeddings) for all non-deleted faces of an image.

        ``face_ids`` contains every non-deleted face ID.  ``embeddings`` is a
        subset: faces absent from FAISS (e.g. after a partial failure) are
        silently omitted so they are skipped during FAISS restore on rollback.
        """
        face_ids: list[int] = list(
            session.execute(
                select(Face.id).where(
                    Face.image_id == image_id,
                    Face.deleted_at.is_(None),
                )
            ).scalars()
        )
        embeddings: dict[int, np.ndarray] = {}
        for fid in face_ids:
            try:
                embeddings[fid] = self.vector_store.get_embedding(fid)
            except IndexError:
                logger.debug(
                    "Face %d absent from FAISS; skipping restore on rollback", fid
                )
        return face_ids, embeddings

    def _detect_faces_in_savepoint(
        self,
        session,
        image_id: int,
        path: Path,
        embedder: FaceEmbedder,
    ) -> tuple[list[tuple[Face, list]], list[int]]:
        """Detect faces, persist them in a DB savepoint, and add embeddings to FAISS.

        Returns ``(faces_with_bboxes, added_face_ids)``.  On savepoint failure all
        DB rows are rolled back and partially-written FAISS vectors are removed
        before re-raising.
        """
        faces_info = embedder.process_image(path)
        new_faces: list[tuple[Face, list]] = []
        added_face_ids: list[int] = []
        try:
            with session.begin_nested():
                for info in faces_info:
                    bbox = info["bbox"]
                    face = Face(
                        image_id=image_id,
                        bbox_x=bbox[0],
                        bbox_y=bbox[1],
                        bbox_w=bbox[2] - bbox[0],
                        bbox_h=bbox[3] - bbox[1],
                        detection_confidence=info["det_score"],
                        state=FaceState.UNIDENTIFIED,
                        model_version="buffalo_l",
                    )
                    session.add(face)
                    session.flush()
                    self.vector_store.add(face.id, info["embedding"])
                    added_face_ids.append(face.id)
                    new_faces.append((face, bbox))
        except Exception:
            for face_id in added_face_ids:
                self.vector_store.remove(face_id)
            raise
        return new_faces, added_face_ids

    def _soft_delete_faces(
        self,
        session,
        face_ids: list[int],
        embeddings: dict[int, np.ndarray],
    ) -> None:
        """Soft-delete faces in DB and remove their embeddings from FAISS."""
        if not face_ids:
            return
        logger.debug("Soft-deleting %d stale face(s)", len(face_ids))
        for fid in face_ids:
            if fid in embeddings:
                self.vector_store.remove(fid)
        session.execute(
            update(Face)
            .where(Face.id.in_(face_ids))
            .values(deleted_at=datetime.now(timezone.utc))
        )

    def _index_single_image(self, img: Image, session) -> int:
        """Detect faces, persist to DB + FAISS, save thumbnails. Returns face count."""
        path = Path(img.file_path)
        if not path.exists():
            img.file_hash = "MISSING"
            session.commit()
            return 0

        img.file_hash = self._calculate_hash(path)

        # On re-index, pre-existing faces are always stale: new Face rows get new
        # auto-increment IDs, so none of them can match the freshly-detected set.
        existing_face_ids, existing_embeddings = self._fetch_existing_face_data(
            session, img.id
        )
        embedder = self._get_embedder()
        new_faces, added_face_ids = self._detect_faces_in_savepoint(
            session, img.id, path, embedder
        )

        try:
            # Keep stale-face removal inside this try so any failure here also
            # triggers the rollback/FAISS-restore path below.
            self._soft_delete_faces(session, existing_face_ids, existing_embeddings)
            self._extract_exif_metadata(path, img.id, session)
            # Persist FAISS first to minimise DB→FAISS mismatch risk.
            self.vector_store.save(self.paths.faiss_path)
            session.commit()
        except Exception:
            # Downstream failure after the savepoint succeeded: roll back the
            # session (discards new Face rows AND deleted_at updates), remove new
            # FAISS vectors, and restore existing embeddings so FAISS stays
            # consistent with the rolled-back DB state.
            session.rollback()
            for face_id in added_face_ids:
                self.vector_store.remove(face_id)
            for fid, embedding in existing_embeddings.items():
                self.vector_store.add(fid, embedding)
            raise

        try:
            self._save_face_crops(new_faces, embedder, path)
        except Exception:
            logger.warning(
                "Face crop saving failed for %s; crops may be missing",
                path,
                exc_info=True,
            )
        return len(new_faces)

    def _index_image_safely(
        self,
        session,
        img: Image,
        indexed_count: int,
        total_images: int,
        total_faces: int,
    ) -> tuple[int, int]:
        """Index one image, emit progress, and return updated counters.

        On failure the image is marked as ERROR and the original counters are returned.
        """
        try:
            new_face_count = self._index_single_image(img, session)
            indexed_count += 1
            total_faces += new_face_count
            status_msg = QtCore.QCoreApplication.translate(
                "IndexingTask", "Indexing photos..."
            )
            self.progress.emit(indexed_count, total_images, total_faces, status_msg)
        except Exception:
            logger.warning("Error indexing %s", img.file_path, exc_info=True)
            session.rollback()
            img.file_hash = "ERROR"
            session.commit()
        return indexed_count, total_faces

    def run(self) -> None:
        """Index all images that haven't been indexed yet."""
        with self.session_factory() as session:
            total_images = session.execute(
                select(func.count(Image.id)).where(Image.file_hash.is_(None))
            ).scalar_one()

            if total_images == 0:
                self.finished.emit()
                return

            total_faces = session.execute(select(func.count(Face.id))).scalar_one()
            indexed_count = 0

            while not self._is_cancelled:
                batch = (
                    session.execute(
                        select(Image).where(Image.file_hash.is_(None)).limit(10)
                    )
                    .scalars()
                    .all()
                )
                if not batch:
                    break

                for img in batch:
                    if self._is_cancelled:
                        break
                    indexed_count, total_faces = self._index_image_safely(
                        session, img, indexed_count, total_images, total_faces
                    )

            self.vector_store.save(self.paths.faiss_path)
            self.finished.emit()
