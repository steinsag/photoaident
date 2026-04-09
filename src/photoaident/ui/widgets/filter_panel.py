from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select

from photoaident.core.date_range import DateRange
from photoaident.core.geo import GpsBoundingBox
from photoaident.db.database import Person
from photoaident.ui.widgets.date_filter_dialog import (
    DateFilterDialog,
    format_date_range,
)
from photoaident.ui.widgets.map_dialog import MapLocationDialog
from photoaident.utils.resource_path import get_resource_path

ASPECT_RATIO_WORLD_MAP_ICON = 1.97
FILTER_PANEL_WIDTH = 220

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.paths import AppPaths


class FilterPanel(QtWidgets.QFrame):
    """Right-side filter panel with location, time, and person filtering."""

    location_changed = QtCore.Signal(object)  # GpsBoundingBox | None
    date_range_changed = QtCore.Signal(object)  # DateRange | None
    person_selection_changed = QtCore.Signal()

    def __init__(
        self,
        session_factory: "sessionmaker",
        paths: "AppPaths",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session_factory = session_factory
        self._paths = paths
        self._gps_bbox: GpsBoundingBox | None = None
        self._date_range: DateRange | None = None

        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.setFixedWidth(FILTER_PANEL_WIDTH)
        panel_layout = QtWidgets.QVBoxLayout(self)

        # --- Location section ---
        self._add_header(self.tr("Location"), panel_layout)

        self.map_location_btn = QtWidgets.QToolButton()
        self.map_location_btn.setText(self.tr("Click to set location"))
        self.map_location_btn.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )
        icon_path = get_resource_path("assets/icons/world_map.svg")
        world_map_icon = QtGui.QIcon(icon_path)
        self.map_location_btn.setIcon(world_map_icon)
        margins = panel_layout.contentsMargins()
        icon_w = FILTER_PANEL_WIDTH - margins.left() - margins.right()
        self.map_location_btn.setIconSize(
            QtCore.QSize(icon_w, round(icon_w / ASPECT_RATIO_WORLD_MAP_ICON))
        )
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

        # --- Time section ---
        self._add_section_separator(panel_layout)
        self._add_header(self.tr("Time"), panel_layout)

        self.date_filter_btn = QtWidgets.QToolButton()
        self.date_filter_btn.setText(self.tr("Click to set time range"))
        self.date_filter_btn.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )
        calendar_icon_path = get_resource_path("assets/icons/calendar.svg")
        calendar_icon = QtGui.QIcon(calendar_icon_path)
        self.date_filter_btn.setIcon(calendar_icon)
        self.date_filter_btn.setIconSize(
            QtCore.QSize(round(icon_w / 3), round(icon_w / 3))
        )
        self.date_filter_btn.setCheckable(True)
        self.date_filter_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.date_filter_btn.clicked.connect(self._open_date_dialog)
        panel_layout.addWidget(self.date_filter_btn)

        self.clear_time_btn = QtWidgets.QPushButton(self.tr("Clear Time"))
        self.clear_time_btn.clicked.connect(self._on_time_cleared)
        self.clear_time_btn.setVisible(False)
        panel_layout.addWidget(self.clear_time_btn)

        # --- Person section ---
        self._add_section_separator(panel_layout)
        self._add_header(self.tr("Person"), panel_layout)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText(self.tr("Type to filter"))
        self.search_edit.textChanged.connect(self._on_search_changed)
        panel_layout.addWidget(self.search_edit)

        self.person_list_widget = QtWidgets.QListWidget()
        self.person_list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self.person_list_widget.itemSelectionChanged.connect(
            self.person_selection_changed
        )
        panel_layout.addWidget(self.person_list_widget)

        self._populate_person_list()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._populate_person_list()

    def gps_bbox(self) -> GpsBoundingBox | None:
        """Return the currently active GPS bounding box, or None."""
        return self._gps_bbox

    def date_range(self) -> DateRange | None:
        """Return the currently active date range filter, or None."""
        return self._date_range

    def selected_person_ids(self) -> list[int]:
        """Return the IDs of all currently selected persons."""
        return [
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self.person_list_widget.selectedItems()
        ]

    def clear_all_filters(self) -> None:
        """Clear all active filters and update UI without emitting change signals."""
        self._gps_bbox = None
        self._update_map_button()
        self._date_range = None
        self._update_time_button()
        self.search_edit.clear()
        self.person_list_widget.blockSignals(True)
        self.person_list_widget.clearSelection()
        self.person_list_widget.blockSignals(False)

    def _populate_person_list(self) -> None:
        """Reload the person list from the database, preserving current selection."""
        selected_ids = set(self.selected_person_ids())
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

    def _open_map_dialog(self) -> None:
        dialog = MapLocationDialog(
            paths=self._paths, initial_bbox=self._gps_bbox, parent=self
        )
        result = dialog.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self._gps_bbox = dialog.selected_bbox()
            self.location_changed.emit(self._gps_bbox)
        self._update_map_button()

    def _on_location_cleared(self) -> None:
        self._gps_bbox = None
        self._update_map_button()
        self.location_changed.emit(None)

    def _update_map_button(self) -> None:
        if self._gps_bbox is None:
            self.clear_location_btn.setVisible(False)
            self.map_location_btn.setChecked(False)
        else:
            self.clear_location_btn.setVisible(True)
            self.map_location_btn.setChecked(True)

    def _open_date_dialog(self) -> None:
        dialog = DateFilterDialog(
            session_factory=self.session_factory,
            initial_range=self._date_range,
            parent=self,
        )
        result = dialog.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self._date_range = dialog.selected_range()
            self.date_range_changed.emit(self._date_range)
        self._update_time_button()

    def _on_time_cleared(self) -> None:
        self._date_range = None
        self._update_time_button()
        self.date_range_changed.emit(None)

    def _update_time_button(self) -> None:
        if self._date_range is None:
            self.clear_time_btn.setVisible(False)
            self.date_filter_btn.setChecked(False)
            self.date_filter_btn.setText(self.tr("Click to set time range"))
        else:
            self.clear_time_btn.setVisible(True)
            self.date_filter_btn.setChecked(True)
            self.date_filter_btn.setText(format_date_range(self._date_range))

    @staticmethod
    def _add_section_separator(layout: QtWidgets.QVBoxLayout) -> None:
        layout.addSpacing(4)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        layout.addSpacing(4)

    @staticmethod
    def _add_header(title: str, panel_widget: QtWidgets.QVBoxLayout) -> None:
        header = QtWidgets.QLabel(title)
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        panel_widget.addWidget(header)
