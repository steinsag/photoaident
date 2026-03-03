from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select

from photoaident.core.geo import GpsBoundingBox
from photoaident.core.search import search_images
from photoaident.db.database import Person
from photoaident.ui.widgets.map_dialog import MapLocationDialog
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
        from photoaident.app import get_resource_path

        self.map_location_btn = QtWidgets.QToolButton()
        self.map_location_btn.setText(self.tr("Click to set location"))
        self.map_location_btn.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )
        icon_path = get_resource_path("assets/icons/world_map.svg")
        self.map_location_btn.setIcon(QtGui.QIcon(icon_path))
        self.map_location_btn.setIconSize(QtCore.QSize(350, 90))
        self.map_location_btn.setCheckable(True)
        self.map_location_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.map_location_btn.clicked.connect(self._open_map_dialog)
        panel_layout.addWidget(self.map_location_btn)

        self.clear_location_btn = QtWidgets.QPushButton(self.tr("Clear Location"))
        self.clear_location_btn.clicked.connect(self._on_location_cleared)
        self.clear_location_btn.setVisible(False)
        panel_layout.addWidget(self.clear_location_btn)

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

    def _has_filters(self) -> bool:
        """Check if any filters (person or location) are active."""
        return bool(self._selected_person_ids()) or self._gps_bbox is not None

    def _open_map_dialog(self) -> None:
        dialog = MapLocationDialog(initial_bbox=self._gps_bbox, parent=self)
        result = dialog.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self._gps_bbox = dialog.selected_bbox()
            self.load_images()
        self._update_map_button()

    def _on_location_cleared(self) -> None:
        self._gps_bbox = None
        self._update_map_button()
        self.load_images()

    def _update_map_button(self) -> None:
        if self._gps_bbox is None:
            self.clear_location_btn.setVisible(False)
            self.map_location_btn.setChecked(False)
        else:
            self.clear_location_btn.setVisible(True)
            self.map_location_btn.setChecked(True)

    def load_images(self) -> None:
        """Fetch search results and update the UI accordingly."""
        if not self._has_filters():
            self.grid.clear()
            self.grid.setVisible(False)
            self.empty_label.setVisible(True)
            self.empty_label.show()  # Explicit show
            return

        person_ids = self._selected_person_ids()
        results = search_images(
            thumbs_dir=self.paths.thumbs_dir,
            session_factory=self.session_factory,
            vector_store=self.vector_store,
            person_ids=person_ids,
            gps_bbox=self._gps_bbox,
        )

        # Update visibility after retrieving results
        if not results:
            self.grid.clear()
        else:
            self.grid.set_results(results)
        self.empty_label.setVisible(False)
        self.empty_label.hide()  # Explicit hide
        self.grid.setVisible(True)
        self.grid.show()  # Explicit show

    def _on_navigate_to_labelling(self, image_id: int) -> None:
        from photoaident.app import MainWindow  # local import breaks circular dep

        main = self.window()
        if isinstance(main, MainWindow):
            main.go_to_labelling(image_id)
