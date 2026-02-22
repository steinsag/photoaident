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

        # 1. Recursive scan for .jpg and .jpeg (case-insensitive)
        image_paths: List[Path] = []
        extensions = {".jpg", ".jpeg"}

        for root, _, files in os.walk(self.root_path):
            if self._is_cancelled:
                self.finished.emit(0)
                return
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    image_paths.append(Path(root) / file)

        total = len(image_paths)
        if total == 0:
            self.finished.emit(0)
            return

        self.status.emit(
            QtCore.QCoreApplication.translate("InventoryTask", "Adding to database...")
        )

        # 2. Add to database in batches
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
                        # Check if already exists by file_path
                        existing = session.execute(
                            select(Image).where(Image.file_path == str(p))
                        ).scalar_one_or_none()
                        if existing:
                            continue

                        # Get file size without opening file
                        stat = p.stat()
                        img = Image(
                            file_path=str(p),
                            file_size=stat.st_size,
                            file_hash=None,  # Defer hashing
                        )
                        session.add(img)
                        added_count += 1
                    except Exception:
                        # Skip files that can't be stat'ed (permissions etc)
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

    def run(self):
        """Index images that haven't been indexed yet."""
        with self.session_factory() as session:
            # 1. Get total count of images to index
            total_images = session.execute(
                select(func.count(Image.id)).where(Image.file_hash.is_(None))
            ).scalar_one()

            if total_images == 0:
                self.finished.emit()
                return

            # Get total face count so far
            total_faces = session.execute(select(func.count(Face.id))).scalar_one()

            # 2. Process images one by one
            indexed_count = 0
            # Get images in smaller batches to avoid long-running transactions
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
                        path = Path(img.file_path)
                        if not path.exists():
                            # Skip missing files
                            img.file_hash = "MISSING"
                            session.commit()
                            continue

                        # Calculate hash
                        img.file_hash = self._calculate_hash(path)

                        # Detect faces
                        embedder = self._get_embedder()
                        faces_info = embedder.process_image(path)

                        new_faces = []
                        for info in faces_info:
                            # Add to FAISS
                            faiss_id = self.vector_store.add(info["embedding"])

                            # Prepare Face for DB
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

                        # First, persist FAISS to minimize DB→FAISS mismatch risk
                        self.vector_store.save(self.paths.faiss_path)

                        # Then commit DB so faces and file_hash are durable
                        session.commit()  # Now face IDs are assigned

                        # Save face crop thumbnails (cache)
                        self.paths.face_crops_dir.mkdir(parents=True, exist_ok=True)
                        for face, bbox in new_faces:
                            crop = embedder.extract_face_crop(path, bbox)
                            crop_path = self.paths.face_crops_dir / f"{face.id}.jpg"
                            crop.save(crop_path, "JPEG", quality=90)

                        # Save full-photo thumbnail (cache)
                        thumb_path = self.paths.thumbs_dir / f"{img.file_hash}.jpg"
                        if not thumb_path.exists():
                            generate_thumbnail(path, thumb_path)

                        indexed_count += 1
                        total_faces += len(faces_info)
                        self.progress.emit(indexed_count, total_images, total_faces)

                    except Exception as e:
                        print(f"Error indexing {img.file_path}: {e}")
                        # Mark as failed somehow?
                        img.file_hash = "ERROR"
                        session.commit()

            # Final save of FAISS index before finishing
            self.vector_store.save(self.paths.faiss_path)
            self.finished.emit()
