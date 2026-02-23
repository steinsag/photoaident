import hashlib
import os
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore
from sqlalchemy import select, func
from sqlalchemy.orm import sessionmaker

from photoaident.core.embeddings import FaceEmbedder
from photoaident.db.database import Image, Face, FaceState
from photoaident.db.vector_store import VectorStore
from photoaident.paths import AppPaths
from photoaident.utils.image_utils import generate_thumbnail


class InventoryTask(QtCore.QObject):
    """
    Background task to scan a directory for images and add them to the database.
    Does not open image files, just inventories paths and sizes.
    """

    progress = QtCore.Signal(int, int)  # current, total
    finished = QtCore.Signal(int)  # total added
    status = QtCore.Signal(str)  # status message

    def __init__(self, root_path: str, session_factory: sessionmaker):
        super().__init__()
        self.root_path = Path(root_path)
        self.session_factory = session_factory
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _scan_image_files(self, extensions: set) -> Optional[List[Path]]:
        """Walk root_path for matching files. Returns None if cancelled."""
        image_paths: List[Path] = []
        for root, _, files in os.walk(self.root_path):
            if self._is_cancelled:
                return None
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    image_paths.append(Path(root) / file)
        return image_paths

    def _add_image_if_missing(self, session, p: Path) -> bool:
        """Insert image record if not already present. Returns True if added."""
        existing = session.execute(
            select(Image).where(Image.file_path == str(p))
        ).scalar_one_or_none()
        if existing:
            return False
        stat = p.stat()
        img = Image(
            file_path=str(p),
            file_size=stat.st_size,
            file_hash=None,
        )
        session.add(img)
        return True

    def run(self):
        """Perform the scan and inventory."""
        if not self.root_path.exists() or not self.root_path.is_dir():
            self.finished.emit(0)
            return

        self.status.emit(
            QtCore.QCoreApplication.translate(
                "InventoryTask", "Searching for photos..."
            )
        )

        image_paths = self._scan_image_files({".jpg", ".jpeg"})
        if image_paths is None:
            self.finished.emit(0)
            return

        total = len(image_paths)
        if total == 0:
            self.finished.emit(0)
            return

        self.status.emit(
            QtCore.QCoreApplication.translate("InventoryTask", "Adding to database...")
        )

        added_count = 0
        batch_size = 100

        with self.session_factory() as session:
            for i in range(0, total, batch_size):
                if self._is_cancelled:
                    session.rollback()
                    self.finished.emit(0)
                    return

                batch = image_paths[i : i + batch_size]
                for p in batch:
                    try:
                        if self._add_image_if_missing(session, p):
                            added_count += 1
                    except Exception:
                        continue

                session.commit()
                self.progress.emit(added_count, total)

        self.finished.emit(added_count)


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

        # Persist FAISS first to minimise DB→FAISS mismatch risk
        self.vector_store.save(self.paths.faiss_path)
        session.commit()

        self._save_face_crops(new_faces, embedder, path)

        thumb_path = self.paths.thumbs_dir / f"{img.file_hash}.jpg"
        if not thumb_path.exists():
            generate_thumbnail(path, thumb_path)

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
                    except Exception as e:
                        print(f"Error indexing {img.file_path}: {e}")
                        img.file_hash = "ERROR"
                        session.commit()

            self.vector_store.save(self.paths.faiss_path)
            self.finished.emit()
