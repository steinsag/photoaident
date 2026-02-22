import os
from pathlib import Path
from typing import List

from PySide6 import QtCore
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
