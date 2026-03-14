import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QKeyEvent
from sqlalchemy import select

from photoaident.core.search import SearchResult
from photoaident.db.database import Image
from photoaident.ui.widgets.thumbnail_grid import ThumbnailGrid

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore
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
        vector_store: "VectorStore",
        parent=None,
    ):
        super().__init__(parent)
        self.session_factory = session_factory
        self.paths = paths
        self.settings = settings
        self.vector_store = vector_store

        self._columns: list[QtWidgets.QListWidget] = []
        self._selected_path: Path | None = None
        self._current_root: Path | None = None

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
        self.grid = ThumbnailGrid(self.session_factory, self.vector_store, self.paths)
        self.grid.navigate_to_labelling.connect(self._on_navigate_to_labelling)
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
        """Called when this page becomes active. Reloads from collection root.

        Skips rebuilding when the collection root hasn't changed and columns
        already exist, preserving the user's folder position across page switches.
        """
        collection_path = self.settings.collection_path
        if not collection_path:
            self._clear_columns()
            self.grid.set_results([])
            self._hint_label.setVisible(True)
            self._current_root = None
            return

        root = Path(collection_path)
        if not root.exists():
            self._clear_columns()
            self.grid.set_results([])
            self._hint_label.setVisible(True)
            self._current_root = None
            return

        self._hint_label.setVisible(False)

        # If same root and columns already built, keep current position but
        # refresh the grid in case new images were indexed since last visit.
        if self._current_root == root and self._columns:
            folder = self._selected_path or root
            self._load_images_for_folder(folder)
            self._columns[-1].setFocus()
            return

        self._current_root = root
        self._rebuild_root_column(root)
        if self._columns:
            self._columns[0].setFocus()

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
        col.installEventFilter(self)
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

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Intercept arrow keys in column widgets for keyboard navigation."""
        if event.type() != QtCore.QEvent.Type.KeyPress:
            return super().eventFilter(obj, event)

        col_index = next((i for i, c in enumerate(self._columns) if c is obj), -1)
        if col_index < 0:
            return super().eventFilter(obj, event)

        col = self._columns[col_index]
        key = cast(QKeyEvent, event).key()

        if key == QtCore.Qt.Key.Key_Left:
            self._navigate_left(col_index)
            return True

        if key == QtCore.Qt.Key.Key_Right:
            self._navigate_right(col_index)
            return True

        if key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):
            prev_row = col.currentRow()
            col.keyPressEvent(cast(QKeyEvent, event))
            if col.currentRow() != prev_row:
                current = col.currentItem()
                if current is not None:
                    self._on_column_item_changed(col_index, current)
            return True

        return super().eventFilter(obj, event)

    def _navigate_left(self, col_index: int) -> None:
        """Move focus one column left and collapse columns to the right."""
        if col_index == 0:
            return
        left_col = self._columns[col_index - 1]
        current = left_col.currentItem()
        if current is not None:
            # _on_column_item_changed calls _add_column → insertWidget, which
            # can transfer focus to the newly inserted widget.  Set focus after
            # the layout mutation so the intended column always wins.
            self._on_column_item_changed(col_index - 1, current)
        left_col.setFocus()

    def _navigate_right(self, col_index: int) -> None:
        """Move focus one column right, selecting its first item if needed."""
        if col_index >= len(self._columns) - 1:
            return
        right_col = self._columns[col_index + 1]
        if right_col.currentItem() is None:
            right_col.setCurrentRow(0)
        current = right_col.currentItem()
        if current is not None:
            # Same reasoning as _navigate_left: set focus after the layout
            # mutation so insertWidget cannot steal it from right_col.
            self._on_column_item_changed(col_index + 1, current)
        right_col.setFocus()

    def _on_navigate_to_labelling(self, image_id: int) -> None:
        from photoaident.app import MainWindow  # local import breaks circular dep

        main = self.window()
        if isinstance(main, MainWindow):
            main.go_to_labelling(image_id)

    def _build_images_data(self, images: list[Image]) -> list[SearchResult]:
        """Build SearchResult objects for ThumbnailGrid."""
        return [
            SearchResult(
                image_id=img.id,
                file_path=img.file_path,
                thumb_path=self.paths.thumbs_dir
                / (f"{img.file_hash}.jpg" if img.file_hash else "unknown.jpg"),
            )
            for img in images
        ]
