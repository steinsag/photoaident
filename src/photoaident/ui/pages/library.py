from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from photoaident.core.search import search_images
from photoaident.ui.widgets.filter_panel import FilterPanel
from photoaident.ui.widgets.thumbnail_grid import ThumbnailGrid

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore
    from photoaident.paths import AppPaths


class LibraryPage(QtWidgets.QWidget):
    """Page showing all indexed images with filtering by person, location, and date."""

    navigate_to_labelling = QtCore.Signal(int)
    navigate_to_browse = QtCore.Signal(str)  # file_path

    def __init__(
        self,
        session_factory: "sessionmaker",
        paths: "AppPaths",
        vector_store: "VectorStore",
        parent=None,
    ):
        super().__init__(parent)
        self.session_factory = session_factory
        self._paths = paths
        self.vector_store = vector_store

        # Top-level horizontal layout: center area + right filter panel
        layout = QtWidgets.QHBoxLayout(self)

        # --- Center area ---
        center_area = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_area)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # File path/name search with explicit Search button
        search_bar_layout = QtWidgets.QHBoxLayout()
        self.filepath_search_edit = QtWidgets.QLineEdit()
        self.filepath_search_edit.setPlaceholderText(
            self.tr("Search by file name or path")
        )
        self.search_button = QtWidgets.QPushButton(self.tr("Search"))
        self.search_button.setEnabled(False)
        self.filepath_search_edit.textChanged.connect(
            lambda text: self.search_button.setEnabled(bool(text.strip()))
        )
        self.filepath_search_edit.textChanged.connect(self._update_reset_button)
        self.filepath_search_edit.returnPressed.connect(self.search_button.click)
        self.search_button.clicked.connect(self.load_images)
        self.reset_button = QtWidgets.QPushButton(self.tr("Reset"))
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self._reset_all_filters)
        search_bar_layout.addWidget(self.filepath_search_edit)
        search_bar_layout.addWidget(self.search_button)
        search_bar_layout.addWidget(self.reset_button)
        center_layout.addLayout(search_bar_layout)

        # Image grid
        self.grid = ThumbnailGrid(self.session_factory, self.vector_store, self._paths)
        self.grid.navigate_to_labelling.connect(self.navigate_to_labelling)
        self.grid.navigate_to_browse.connect(self.navigate_to_browse)
        center_layout.addWidget(self.grid, stretch=1)

        # Placeholder shown when no filters are active
        self.empty_label = QtWidgets.QLabel(
            self.tr("Select a person, location, or time range to start searching.")
        )
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setVisible(False)
        center_layout.addWidget(self.empty_label, stretch=1)

        layout.addWidget(center_area, stretch=1)

        # --- Right filter panel (permanent, always visible) ---
        self.filter_panel = FilterPanel(session_factory, paths, parent=self)
        self.filter_panel.location_changed.connect(lambda _: self.load_images())
        self.filter_panel.date_range_changed.connect(lambda _: self.load_images())
        self.filter_panel.person_selection_changed.connect(self.load_images)
        layout.addWidget(self.filter_panel)

        self.load_images()

    def _has_filters(self) -> bool:
        """Check if any filters (person, location, time, or filename) are active."""
        return (
            bool(self.filter_panel.selected_person_ids())
            or self.filter_panel.gps_bbox() is not None
            or self.filter_panel.date_range() is not None
            or bool(self.filepath_search_edit.text().strip())
        )

    def _update_reset_button(self) -> None:
        """Enable or disable the Reset button based on whether any filter is active."""
        self.reset_button.setEnabled(self._has_filters())

    def _reset_all_filters(self) -> None:
        """Clear all active search filters and reload images."""
        self.filter_panel.clear_all_filters()
        self.filepath_search_edit.clear()
        self.load_images()

    def load_images(self) -> None:
        """Fetch search results and update the UI accordingly."""
        self._update_reset_button()

        if not self._has_filters():
            self.grid.clear()
            self.grid.setVisible(False)
            self.empty_label.setVisible(True)
            self.empty_label.show()  # Explicit show
            return

        person_ids = self.filter_panel.selected_person_ids()
        filename_query = self.filepath_search_edit.text().strip() or None
        results = search_images(
            thumbs_dir=self._paths.thumbs_dir,
            session_factory=self.session_factory,
            vector_store=self.vector_store,
            person_ids=person_ids,
            gps_bbox=self.filter_panel.gps_bbox(),
            date_range=self.filter_panel.date_range(),
            filename_query=filename_query,
        )

        self.grid.set_relevance_available(bool(person_ids))

        # Update visibility after retrieving results
        if not results:
            self.grid.clear()
        else:
            self.grid.set_results(results)
        self.empty_label.setVisible(False)
        self.empty_label.hide()  # Explicit hide
        self.grid.setVisible(True)
        self.grid.show()  # Explicit show
