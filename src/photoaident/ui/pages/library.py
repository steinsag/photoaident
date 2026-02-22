from typing import TYPE_CHECKING

from PySide6 import QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from photoaident.db.database import Image
from photoaident.ui.widgets.thumbnail_grid import ThumbnailGrid

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker
    from photoaident.paths import AppPaths


class LibraryPage(QtWidgets.QWidget):
    """Page showing all indexed images with filtering."""

    def __init__(self, session_factory: "sessionmaker", paths: "AppPaths", parent=None):
        super().__init__(parent)
        self.session_factory = session_factory
        self.paths = paths

        layout = QtWidgets.QVBoxLayout(self)

        # Filter bar
        filter_layout = QtWidgets.QHBoxLayout()
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(
            [self.tr("All Images"), self.tr("With Faces"), self.tr("Without Faces")]
        )
        self.filter_combo.currentIndexChanged.connect(self.load_images)
        filter_layout.addWidget(QtWidgets.QLabel(self.tr("Filter:")))
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()

        layout.addLayout(filter_layout)

        # Image grid
        self.grid = ThumbnailGrid()
        layout.addWidget(self.grid)

        # Initial load
        self.load_images()

    def load_images(self):
        filter_idx = self.filter_combo.currentIndex()

        with self.session_factory() as session:
            stmt = select(Image).options(
                joinedload(Image.faces), joinedload(Image.metadata_rel)
            )

            if filter_idx == 1:  # With Faces
                stmt = stmt.where(Image.faces.any())
            elif filter_idx == 2:  # Without Faces
                stmt = stmt.where(~Image.faces.any())

            images = session.execute(stmt).unique().scalars().all()

            images_data = []
            for img in images:
                # Calculate thumbnail path
                thumb_path = (
                    self.paths.thumbs_dir / f"{img.file_hash}.jpg"
                    if img.file_hash
                    else self.paths.thumbs_dir / "unknown.jpg"
                )

                # Get dimensions if available
                orig_size = None
                if img.metadata_rel:
                    orig_size = (img.metadata_rel.width, img.metadata_rel.height)

                images_data.append(
                    (img.id, img.file_path, img.faces, thumb_path, orig_size)
                )

            self.grid.set_images(images_data)
