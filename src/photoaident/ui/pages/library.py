from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select

from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search import find_images_by_person, find_images_by_gps_bbox
from photoaident.db.database import Image, Person
from photoaident.ui.widgets.map_dialog import MapLocationDialog
from photoaident.ui.widgets.map_preview import MapPreviewWidget
from photoaident.ui.widgets.thumbnail_grid import ThumbnailGrid

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore
    from photoaident.paths import AppPaths


class LibraryPage(QtWidgets.QWidget):
    """Page showing all indexed images with filtering by person."""

    def __init__(
        self,
        session_factory: "sessionmaker",
        paths: "AppPaths",
        vector_store: "VectorStore | None" = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session_factory = session_factory
        self.paths = paths
        self.vector_store = vector_store
        self._gps_bbox: GpsBoundingBox | None = None

        # Top-level horizontal layout: center area + right filter panel
        layout = QtWidgets.QHBoxLayout(self)

        # --- Center area ---
        center_area = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_area)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # Non-functional keyword search bar
        self.keyword_search_edit = QtWidgets.QLineEdit()
        self.keyword_search_edit.setPlaceholderText(
            self.tr("Type to search by keyword. Use @\u2026 to search for person.")
        )
        center_layout.addWidget(self.keyword_search_edit)

        # Image grid
        self.grid = ThumbnailGrid(self.session_factory)
        self.grid.navigate_to_labelling.connect(self._on_navigate_to_labelling)
        center_layout.addWidget(self.grid, stretch=1)

        # Placeholder shown when no person is selected
        self.empty_label = QtWidgets.QLabel(
            self.tr("Select a person or location to start searching.")
        )
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setVisible(False)
        center_layout.addWidget(self.empty_label, stretch=1)

        layout.addWidget(center_area, stretch=1)

        # --- Right filter panel (permanent, always visible) ---
        self.filter_panel = QtWidgets.QFrame()
        self.filter_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.filter_panel.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.filter_panel.setFixedWidth(220)
        panel_layout = QtWidgets.QVBoxLayout(self.filter_panel)

        self.map_preview = MapPreviewWidget()
        self.map_preview.clicked.connect(self._open_map_dialog)
        self.map_preview.cleared.connect(self._on_location_cleared)
        panel_layout.addWidget(self.map_preview)

        person_header = QtWidgets.QLabel(self.tr("Person"))
        font = person_header.font()
        font.setBold(True)
        person_header.setFont(font)
        panel_layout.addWidget(person_header)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText(self.tr("Type to filter"))
        self.search_edit.textChanged.connect(self._on_search_changed)
        panel_layout.addWidget(self.search_edit)

        self.person_list_widget = QtWidgets.QListWidget()
        self.person_list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self.person_list_widget.itemSelectionChanged.connect(self.load_images)
        panel_layout.addWidget(self.person_list_widget)

        layout.addWidget(self.filter_panel)

        self._populate_person_list()
        self.load_images()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._populate_person_list()

    def _populate_person_list(self) -> None:
        selected_ids = set(self._selected_person_ids())
        self.person_list_widget.blockSignals(True)
        self.person_list_widget.clear()
        with self.session_factory() as session:
            persons = session.scalars(select(Person).order_by(Person.name)).all()
            for person in persons:
                item = QtWidgets.QListWidgetItem(person.name)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, person.id)
                self.person_list_widget.addItem(item)
                if person.id in selected_ids:
                    item.setSelected(True)
        self.person_list_widget.blockSignals(False)
        self._apply_search_filter(self.search_edit.text())

    def _apply_search_filter(self, text: str) -> None:
        lower = text.lower()
        for i in range(self.person_list_widget.count()):
            item = self.person_list_widget.item(i)
            if item is None:
                continue
            hidden = bool(lower) and lower not in item.text().lower()
            item.setHidden(hidden)

    def _on_search_changed(self, text: str) -> None:
        self._apply_search_filter(text)

    def _selected_person_ids(self) -> list[int]:
        return [
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self.person_list_widget.selectedItems()
        ]

    def _open_map_dialog(self) -> None:
        dialog = MapLocationDialog(initial_bbox=self._gps_bbox, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._gps_bbox = dialog.selected_bbox()
            self.map_preview.set_bbox(self._gps_bbox)
            self.load_images()

    def _on_location_cleared(self) -> None:
        self._gps_bbox = None
        self.map_preview.set_bbox(None)
        self.load_images()

    def _build_images_data(self, images: list) -> list:
        result = []
        for img in images:
            thumb_path = (
                self.paths.thumbs_dir / f"{img.file_hash}.jpg"
                if img.file_hash
                else self.paths.thumbs_dir / "unknown.jpg"
            )
            result.append((img.id, img.file_path, thumb_path))
        return result

    def load_images(self) -> None:
        person_ids = self._selected_person_ids()
        gps_image_ids: set[int] | None = None

        if self._gps_bbox:
            gps_image_ids = set(
                find_images_by_gps_bbox(self.session_factory, self._gps_bbox)
            )

        if not person_ids and gps_image_ids is None:
            self.grid.clear()
            self.grid.setVisible(False)
            self.empty_label.setVisible(True)
            self.empty_label.show()  # Explicit show
            return

        # Ensure grid is visible and empty label is hidden if we have any filters
        self.empty_label.setVisible(False)
        self.empty_label.hide()  # Explicit hide
        self.grid.setVisible(True)
        self.grid.show()  # Explicit show

        if not person_ids and gps_image_ids is not None:
            # GPS filter only
            with self.session_factory() as session:
                stmt = select(Image).where(Image.id.in_(list(gps_image_ids)))
                images = session.execute(stmt).unique().scalars().all()
                self.grid.set_results(self._build_images_data(images))
            return

        if self.vector_store is None:
            with self.session_factory() as session:
                stmt = select(Image)
                if gps_image_ids is not None:
                    stmt = stmt.where(Image.id.in_(list(gps_image_ids)))
                images = session.execute(stmt).unique().scalars().all()
                self.grid.set_results(self._build_images_data(images))
            return

        # Intersect FAISS results: only images where ALL selected persons appear
        per_person_scores: list[dict[int, float]] = []
        for person_id in person_ids:
            scores: dict[int, float] = {}
            for img_id, score in find_images_by_person(
                self.session_factory, self.vector_store, person_id
            ):
                if gps_image_ids is None or img_id in gps_image_ids:
                    scores[img_id] = score
            per_person_scores.append(scores)

        if not per_person_scores:
            self.grid.set_results([])
            return

        common_ids = set(per_person_scores[0].keys())
        for scores in per_person_scores[1:]:
            common_ids &= scores.keys()

        # Rank by minimum score across persons (weakest match determines relevance)
        image_scores: dict[int, float] = {
            img_id: min(s[img_id] for s in per_person_scores) for img_id in common_ids
        }

        if not image_scores:
            self.grid.clear()
            return

        sorted_pairs = sorted(image_scores.items(), key=lambda kv: kv[1], reverse=True)
        ordered_ids = [img_id for img_id, _ in sorted_pairs]

        with self.session_factory() as session:
            stmt = select(Image).where(Image.id.in_(ordered_ids))
            images = session.execute(stmt).unique().scalars().all()
            image_map = {img.id: img for img in images}
            ordered_images = [image_map[i] for i in ordered_ids if i in image_map]
            self.grid.set_results(self._build_images_data(ordered_images))

    def _on_navigate_to_labelling(self, image_id: int) -> None:
        from photoaident.app import MainWindow  # local import breaks circular dep

        main = self.window()
        if isinstance(main, MainWindow):
            main.go_to_labelling(image_id)
