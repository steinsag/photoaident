import os
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets
from sqlalchemy import select

from photoaident.db.database import Image
from photoaident.ui.widgets.thumbnail_grid import ThumbnailGrid

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.paths import AppPaths
    from photoaident.settings import Settings


class BrowsePage(QtWidgets.QWidget):
    """Browse page: explore photo collection by folder using Miller columns."""

    COLUMN_WIDTH = 200
    COLUMNS_AREA_HEIGHT = 250

    def __init__(
        self,
        session_factory: "sessionmaker",
        paths: "AppPaths",
        settings: "Settings",
        parent=None,
    ):
        super().__init__(parent)
        self.session_factory = session_factory
        self.paths = paths
        self.settings = settings

        self._columns: list[QtWidgets.QListWidget] = []
        self._selected_path: Path | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Top: scrollable area for Miller columns
        self._scroll_area = QtWidgets.QScrollArea()
        self._scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFixedHeight(self.COLUMNS_AREA_HEIGHT)

        self._columns_container = QtWidgets.QWidget()
        self._columns_layout = QtWidgets.QHBoxLayout(self._columns_container)
        self._columns_layout.setContentsMargins(0, 0, 0, 0)
        self._columns_layout.setSpacing(0)
        self._columns_layout.addStretch()

        self._scroll_area.setWidget(self._columns_container)
        splitter.addWidget(self._scroll_area)

        # Bottom: thumbnail grid
        self.grid = ThumbnailGrid()
        splitter.addWidget(self.grid)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Hint label shown when no collection is configured
        self._hint_label = QtWidgets.QLabel(self.tr("No photo collection configured."))
        self._hint_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setVisible(False)
        layout.addWidget(self._hint_label)

    def refresh(self) -> None:
        """Called when this page becomes active. Reloads from collection root."""
        collection_path = self.settings.collection_path
        if not collection_path:
            self._clear_columns()
            self.grid.set_results([])
            self._hint_label.setVisible(True)
            return

        root = Path(collection_path)
        if not root.exists():
            self._clear_columns()
            self.grid.set_results([])
            self._hint_label.setVisible(True)
            return

        self._hint_label.setVisible(False)
        self._rebuild_root_column(root)

    def _clear_columns(self) -> None:
        """Remove all column widgets from the layout."""
        # Remove stretch first
        stretch_item = self._columns_layout.takeAt(self._columns_layout.count() - 1)
        del stretch_item

        for col in self._columns:
            self._columns_layout.removeWidget(col)
            col.deleteLater()
        self._columns.clear()

        # Re-add stretch
        self._columns_layout.addStretch()
        self._update_container_width()

    def _rebuild_root_column(self, root: Path) -> None:
        """Clear all columns, place root itself in column 0, and auto-select it.

        Column 0 always contains the collection root folder as a single selectable
        item.  Selecting it loads root-level direct images and reveals its
        subfolders in column 1 — giving the user a persistent "home" entry to
        navigate back to.
        """
        self._clear_columns()
        self._selected_path = None
        self.grid.set_results([])

        # Column 0: the root folder itself (single item, always present)
        self._add_column([root])

        # Set visual selection and trigger initial content load directly.
        # We cannot rely on itemClicked here (no mouse event), so we call the
        # handler ourselves after marking the row as current.
        if self._columns:
            self._columns[0].setCurrentRow(0)
            item = self._columns[0].item(0)
            if item is not None:
                self._on_column_item_changed(0, item)

    def _add_column(self, folders: list[Path]) -> None:
        """Append a new QListWidget column populated with the given folders."""
        col_index = len(self._columns)

        col = QtWidgets.QListWidget()
        col.setFixedWidth(self.COLUMN_WIDTH)
        col.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        for folder in folders:
            item = QtWidgets.QListWidgetItem(folder.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, folder)
            col.addItem(item)

        # Insert before the trailing stretch
        insert_pos = self._columns_layout.count() - 1
        self._columns_layout.insertWidget(insert_pos, col)
        self._columns.append(col)
        self._update_container_width()

        # Scroll to make the new column fully visible.  The scrollbar maximum is
        # not updated until Qt reflows the layout, so defer by one event-loop tick.
        QtCore.QTimer.singleShot(0, self._scroll_to_end)

        # itemClicked fires on every click, including re-clicks on the already-
        # selected item — which is the correct Finder/Miller-column behaviour.
        col.itemClicked.connect(
            lambda item, idx=col_index: self._on_column_item_changed(idx, item)
        )

    def _update_container_width(self) -> None:
        """Resize the columns container so the scroll area scrolls correctly."""
        total = len(self._columns) * self.COLUMN_WIDTH
        self._columns_container.setMinimumWidth(total)

    def _scroll_to_end(self) -> None:
        """Scroll the columns area so the rightmost column is fully visible."""
        sb = self._scroll_area.horizontalScrollBar()
        sb.setValue(sb.maximum())

    def _on_column_item_changed(
        self, col_index: int, current: QtWidgets.QListWidgetItem | None
    ) -> None:
        """Handle a folder selection in column col_index."""
        if current is None:
            return

        # Remove all columns to the right of col_index
        while len(self._columns) > col_index + 1:
            col = self._columns.pop()
            self._columns_layout.removeWidget(col)
            col.deleteLater()

        self._update_container_width()

        folder: Path = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._selected_path = folder
        self._load_images_for_folder(folder)

        subfolders = self._get_subfolders(folder)
        if subfolders:
            self._add_column(subfolders)

    def _get_subfolders(self, path: Path) -> list[Path]:
        """Return sorted list of subdirectories under path."""
        try:
            return sorted(
                (p for p in path.iterdir() if p.is_dir()),
                key=lambda p: p.name.lower(),
            )
        except PermissionError:
            return []

    def _load_images_for_folder(self, folder: Path) -> None:
        """Query DB for images whose direct parent is folder, update grid."""
        prefix = str(folder) + os.sep
        with self.session_factory() as session:
            stmt = select(Image).where(Image.file_path.like(f"{prefix}%"))
            images = session.scalars(stmt).all()
            direct = [img for img in images if Path(img.file_path).parent == folder]
            self.grid.set_results(self._build_images_data(direct))

    def _build_images_data(self, images: list) -> list:
        """Build (image_id, file_path, thumb_path) tuples for ThumbnailGrid."""
        result = []
        for img in images:
            thumb_path = (
                self.paths.thumbs_dir / f"{img.file_hash}.jpg"
                if img.file_hash
                else self.paths.thumbs_dir / "unknown.jpg"
            )
            result.append((img.id, img.file_path, thumb_path))
        return result
