import logging
import math
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets, QtQuickWidgets

from photoaident.core.geo import GpsBoundingBox
from photoaident.paths import AppPaths

logger = logging.getLogger(__name__)


class MapLocationDialog(QtWidgets.QDialog):
    """Dialog for selecting a GPS bounding box using an interactive map.

    Users can pan and zoom the map to position their area of interest within
    a fixed rectangle overlay. The bounding box corresponding to this rectangle
    is extracted when the dialog is accepted.
    """

    def __init__(
        self,
        paths: AppPaths,
        initial_bbox: Optional[GpsBoundingBox] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Select Location"))
        self.resize(1024, 768)

        self._selected_bbox: Optional[GpsBoundingBox] = None
        self._paths = paths

        self._setup_ui(initial_bbox)

    def _setup_ui(self, initial_bbox: Optional[GpsBoundingBox] = None) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._setup_instruction_label(layout)
        self._setup_map_widget(layout, initial_bbox)
        self._setup_button_box(layout)

    def _setup_instruction_label(self, layout: QtWidgets.QVBoxLayout) -> None:
        instruction = QtWidgets.QLabel(
            self.tr(
                "Pan and zoom the map. The highlighted area defines the search region."
            )
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

    def _setup_map_widget(
        self, layout: QtWidgets.QVBoxLayout, initial_bbox: Optional[GpsBoundingBox]
    ) -> None:
        # Setup QQuickWidget for the QML map
        self._quick_widget = QtQuickWidgets.QQuickWidget()
        self._quick_widget.setResizeMode(
            QtQuickWidgets.QQuickWidget.ResizeMode.SizeRootObjectToView
        )
        # Strong focus so the widget receives mouse-wheel events from the host.
        self._quick_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # Expanding policy + stretch so the map fills all available vertical space.
        self._quick_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        qml_path = Path(__file__).parent / "map_view.qml"
        self._quick_widget.setSource(QtCore.QUrl.fromLocalFile(str(qml_path)))
        self._log_qml_errors()

        root_obj = self._quick_widget.rootObject()
        if root_obj:
            # Create the directory before QML tries to write to it.
            self._paths.tiles_dir.mkdir(parents=True, exist_ok=True)
            root_obj.setProperty("cachePath", str(self._paths.tiles_dir))
            if initial_bbox:
                self._apply_initial_bbox(root_obj, initial_bbox)

        layout.addWidget(self._quick_widget, stretch=1)

    def _log_qml_errors(self) -> None:
        """Log any errors that occurred while loading the QML source."""
        if self._quick_widget.status() == QtQuickWidgets.QQuickWidget.Status.Error:
            for err in self._quick_widget.errors():
                logger.error("QML error: %s", err)

    @staticmethod
    def _apply_initial_bbox(root_obj: object, initial_bbox: GpsBoundingBox) -> None:
        """Set the initial map center and zoom level from a bounding box."""
        center_lat = (initial_bbox.south + initial_bbox.north) / 2
        center_lon = (initial_bbox.west + initial_bbox.east) / 2
        lat_span = initial_bbox.north - initial_bbox.south
        lon_span = abs(initial_bbox.east - initial_bbox.west)
        span = max(lat_span, lon_span)
        if span > 0:
            # Derive zoom so that the 70% selection rect covers the bbox.
            # At zoom z the selection rect spans ~252/2^z degrees.
            zoom = int(max(2, min(15, round(math.log2(252.0 / span)))))
        else:
            zoom = 14
        root_obj.setProperty("initialLat", center_lat)  # type: ignore[attr-defined]
        root_obj.setProperty("initialLon", center_lon)  # type: ignore[attr-defined]
        root_obj.setProperty("initialZoom", zoom)  # type: ignore[attr-defined]

    def _setup_button_box(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    @staticmethod
    def _build_bbox(
        south: float, west: float, north: float, east: float
    ) -> Optional[GpsBoundingBox]:
        """Build a GpsBoundingBox from raw coordinate values."""
        try:
            return GpsBoundingBox(
                south=float(south),
                west=float(west),
                north=float(north),
                east=float(east),
            )
        except (TypeError, ValueError):
            return None

    def _extract_bbox(self) -> Optional[GpsBoundingBox]:
        """Read current bbox values from the QML root object."""
        root_obj = self._quick_widget.rootObject()
        if not root_obj:
            return None
        return self._build_bbox(
            root_obj.property("south"),
            root_obj.property("west"),
            root_obj.property("north"),
            root_obj.property("east"),
        )

    def _on_accept(self) -> None:
        self._selected_bbox = self._extract_bbox()
        self.accept()

    def selected_bbox(self) -> Optional[GpsBoundingBox]:
        """Return the selected GpsBoundingBox, or None if canceled."""
        return self._selected_bbox
