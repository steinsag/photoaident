import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import exifread
from PIL import Image as PILImage
from PySide6 import QtCore
from sqlalchemy import select, func
from sqlalchemy.orm import sessionmaker

from photoaident.core.embeddings import FaceEmbedder
from photoaident.db.database import Image, Face, FaceState, ImageMetadata, TakenAtSource
from photoaident.db.vector_store import VectorStore
from photoaident.paths import AppPaths

logger = logging.getLogger(__name__)


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


class IndexingTask(QtCore.QObject):
    """
    Background task to compute embeddings and detect faces in images.
    Updates the database and FAISS index continuously.
    """

    progress = QtCore.Signal(int, int, int)  # indexed_images, total_images, total_faces
    finished = QtCore.Signal()
    status = QtCore.Signal(str)

    def __init__(
        self,
        session_factory: sessionmaker,
        vector_store: VectorStore,
        paths: AppPaths,
        ctx_id: int = 0,
    ):
        super().__init__()
        self.session_factory = session_factory
        self.vector_store = vector_store
        self.paths = paths
        self.ctx_id = ctx_id
        self._is_cancelled = False
        self._embedder: Optional[FaceEmbedder] = None

    def cancel(self):
        self._is_cancelled = True

    def _get_embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            self._embedder = FaceEmbedder(ctx_id=self.ctx_id)
        return self._embedder

    def _calculate_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_taken_at(self, tags: dict, path: Path) -> tuple[datetime, TakenAtSource]:
        """Determine when the image was taken, falling back to filesystem mtime."""
        for tag_key in (
            "EXIF DateTimeOriginal",
            "EXIF DateTimeDigitized",
            "Image DateTime",
        ):
            if tag_key in tags:
                try:
                    taken_at = datetime.strptime(
                        str(tags[tag_key]), "%Y:%m:%d %H:%M:%S"
                    )
                    return taken_at, TakenAtSource.EXIF
                except ValueError:
                    continue

        taken_at = datetime.fromtimestamp(path.stat().st_mtime)
        return taken_at, TakenAtSource.FILESYSTEM

    def _get_gps_info(
        self, tags: dict
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Extract GPS latitude, longitude, and altitude from EXIF tags."""
        gps_lat = None
        gps_lon = None
        gps_altitude = None

        lat_tag = tags.get("GPS GPSLatitude")
        lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
        lon_tag = tags.get("GPS GPSLongitude")
        lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))

        if lat_tag and lon_tag:
            gps_lat = _dms_to_decimal(lat_tag.values, lat_ref)
            gps_lon = _dms_to_decimal(lon_tag.values, lon_ref)

        alt_tag = tags.get("GPS GPSAltitude")
        alt_ref_tag = tags.get("GPS GPSAltitudeRef")
        if alt_tag:
            try:
                alt_val = float(alt_tag.values[0].num) / float(alt_tag.values[0].den)
                if alt_ref_tag and str(alt_ref_tag) == "1":
                    alt_val = -alt_val
                gps_altitude = alt_val
            except Exception:
                pass

        return gps_lat, gps_lon, gps_altitude

    def _get_camera_and_image_info(
        self, tags: dict
    ) -> tuple[str | None, str | None, int]:
        """Extract camera make, model, and image orientation."""
        camera_make = str(tags["Image Make"]) if "Image Make" in tags else None
        camera_model = str(tags["Image Model"]) if "Image Model" in tags else None

        orientation = 1
        if "Image Orientation" in tags:
            try:
                orientation = int(tags["Image Orientation"].values[0])
            except (ValueError, TypeError, AttributeError):
                pass
        return camera_make, camera_model, orientation

    def _extract_exif_metadata(self, path: Path, image_id: int, session) -> None:
        """Extract EXIF metadata from an image file and persist it to image_metadata."""
        try:
            with PILImage.open(path) as pil_img:
                width, height = pil_img.size

            with open(path, "rb") as f:
                tags = exifread.process_file(f, details=False)

            taken_at, taken_at_source = self._get_taken_at(tags, path)
            gps_lat, gps_lon, gps_altitude = self._get_gps_info(tags)
            camera_make, camera_model, orientation = self._get_camera_and_image_info(
                tags
            )

            metadata = ImageMetadata(
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
            session.add(metadata)
        except Exception:
            logger.warning("EXIF extraction failed for %s", path, exc_info=True)

    def _save_face_crops(
        self,
        new_faces: list,
        embedder: FaceEmbedder,
        image_path: Path,
    ) -> None:
        """Save JPEG crops for each newly detected face."""
        self.paths.face_crops_dir.mkdir(parents=True, exist_ok=True)
        for face, bbox in new_faces:
            crop = embedder.extract_face_crop(image_path, bbox)
            crop_path = self.paths.face_crops_dir / f"{face.id}.jpg"
            crop.save(crop_path, "JPEG", quality=90)

    def _index_single_image(
        self,
        img: Image,
        session,
        embedder: FaceEmbedder,
    ) -> int:
        """Detect faces, persist to DB + FAISS, save thumbnails. Returns face count."""
        path = Path(img.file_path)
        if not path.exists():
            img.file_hash = "MISSING"
            session.commit()
            return 0

        img.file_hash = self._calculate_hash(path)

        faces_info = embedder.process_image(path)
        new_faces = []
        for info in faces_info:
            faiss_id = self.vector_store.add(info["embedding"])
            bbox = info["bbox"]
            face = Face(
                image_id=img.id,
                faiss_id=faiss_id,
                bbox_x=bbox[0],
                bbox_y=bbox[1],
                bbox_w=bbox[2] - bbox[0],
                bbox_h=bbox[3] - bbox[1],
                detection_confidence=info["det_score"],
                state=FaceState.UNIDENTIFIED,
                model_version="buffalo_l",
            )
            session.add(face)
            new_faces.append((face, bbox))

        self._extract_exif_metadata(path, img.id, session)

        # Persist FAISS first to minimise DB→FAISS mismatch risk
        self.vector_store.save(self.paths.faiss_path)
        session.commit()

        self._save_face_crops(new_faces, embedder, path)

        return len(faces_info)

    def run(self):
        """Index images that haven't been indexed yet."""
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
                images_to_index = (
                    session.execute(
                        select(Image).where(Image.file_hash.is_(None)).limit(10)
                    )
                    .scalars()
                    .all()
                )

                if not images_to_index:
                    break

                for img in images_to_index:
                    if self._is_cancelled:
                        break

                    try:
                        embedder = self._get_embedder()
                        new_face_count = self._index_single_image(
                            img, session, embedder
                        )
                        indexed_count += 1
                        total_faces += new_face_count
                        self.progress.emit(indexed_count, total_images, total_faces)
                    except Exception:
                        logger.warning(
                            "Error indexing %s", img.file_path, exc_info=True
                        )
                        img.file_hash = "ERROR"
                        session.commit()

            self.vector_store.save(self.paths.faiss_path)
            self.finished.emit()
