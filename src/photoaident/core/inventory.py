import os
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from photoaident.db.database import Image


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
