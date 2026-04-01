"""Reusable self-contained map widget backed by QML."""

import logging
from pathlib import Path
from typing import Any, Callable

from PySide6 import QtCore, QtGui, QtQuickWidgets, QtWidgets

from photoaident.core.geo import GpsBoundingBox
from photoaident.paths import AppPaths
from photoaident.utils.resource_path import icon_path as _icon_path

logger = logging.getLogger(__name__)


def _apply_initial_bbox(root_obj: QtCore.QObject, bbox: GpsBoundingBox) -> None:
    """Seed the QML map with an initial bounding box.

    Pre-populates extraction properties (south/west/north/east) so that
    current_bbox() returns the correct value even before updateBbox() has run.
    Also triggers the pending-bbox view-fit mechanism via pendingBbox=True.
    """
    for name, value in (
        ("south", bbox.south),
        ("west", bbox.west),
        ("north", bbox.north),
        ("east", bbox.east),
        ("pendingBboxSouth", bbox.south),
        ("pendingBboxWest", bbox.west),
        ("pendingBboxNorth", bbox.north),
        ("pendingBboxEast", bbox.east),
    ):
        root_obj.setProperty(name, value)
    # Toggle pendingBbox false→true so onPendingBboxChanged fires even when a
    # previous call left it true (Qt property change signals only fire on a
    # value change, so setting true while already true is a no-op).
    root_obj.setProperty("pendingBbox", False)
    root_obj.setProperty("pendingBbox", True)


def _build_bbox(south: Any, west: Any, north: Any, east: Any) -> GpsBoundingBox | None:
    """Build a GpsBoundingBox from raw coordinate values, or None on error."""
    try:
        return GpsBoundingBox(
            south=float(south),
            west=float(west),
            north=float(north),
            east=float(east),
        )
    except (TypeError, ValueError):
        return None


class MapWidget(QtWidgets.QWidget):
    """Self-contained map widget with built-in zoom controls.

    Wraps a QQuickWidget running map_view.qml and exposes a clean Python API.
    All QML internals are encapsulated — consumers never access root objects
    or QML properties directly.

    Args:
        paths: Application paths (used for tile cache directory).
        show_overlay: When True the blue selection rectangle is shown and bbox
            extraction is active (MapLocationDialog mode). When False the
            overlay is hidden and a pin marker can be shown instead
            (ImageDetailDialog mode).
        parent: Optional parent widget.
    """

    def __init__(
        self,
        paths: AppPaths,
        show_overlay: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._show_overlay = show_overlay
        self._ready = False
        self._pending_ops: list[Callable[[], None]] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._quick_widget = QtQuickWidgets.QQuickWidget()
        self._quick_widget.setResizeMode(
            QtQuickWidgets.QQuickWidget.ResizeMode.SizeRootObjectToView
        )
        self._quick_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._quick_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self._paths.tiles_dir.mkdir(parents=True, exist_ok=True)
        self._quick_widget.setInitialProperties(
            {
                "cachePath": str(self._paths.tiles_dir),
                "showOverlay": self._show_overlay,
            }
        )

        self._quick_widget.statusChanged.connect(self._on_qml_status_changed)

        qml_path = Path(__file__).parent / "map_view.qml"
        self._quick_widget.setSource(QtCore.QUrl.fromLocalFile(str(qml_path)))

        layout.addWidget(self._quick_widget, stretch=1)
        layout.addLayout(self._build_zoom_buttons())

    def _build_zoom_buttons(self) -> QtWidgets.QHBoxLayout:
        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addStretch()

        self._zoom_out_btn = QtWidgets.QPushButton()
        self._zoom_out_btn.setIcon(QtGui.QIcon(_icon_path("zoom-out.svg")))
        self._zoom_out_btn.setToolTip(self.tr("Zoom out"))
        self._zoom_out_btn.setAccessibleName(self.tr("Zoom out"))
        self._zoom_out_btn.setDefault(False)
        self._zoom_out_btn.setAutoDefault(False)
        self._zoom_out_btn.clicked.connect(self._on_zoom_out)
        zoom_layout.addWidget(self._zoom_out_btn)

        self._zoom_in_btn = QtWidgets.QPushButton()
        self._zoom_in_btn.setIcon(QtGui.QIcon(_icon_path("zoom-in.svg")))
        self._zoom_in_btn.setToolTip(self.tr("Zoom in"))
        self._zoom_in_btn.setAccessibleName(self.tr("Zoom in"))
        self._zoom_in_btn.setDefault(False)
        self._zoom_in_btn.setAutoDefault(False)
        self._zoom_in_btn.clicked.connect(self._on_zoom_in)
        zoom_layout.addWidget(self._zoom_in_btn)

        zoom_layout.addStretch()
        return zoom_layout

    def _on_qml_status_changed(
        self, status: QtQuickWidgets.QQuickWidget.Status
    ) -> None:
        """Handle QML load completion: apply buffered operations or log errors."""
        if status == QtQuickWidgets.QQuickWidget.Status.Error:
            for err in self._quick_widget.errors():
                logger.error("QML error: %s", err)
        elif status == QtQuickWidgets.QQuickWidget.Status.Ready:
            self._ready = True
            for op in self._pending_ops:
                op()
            self._pending_ops.clear()

    def _root_object(self) -> QtCore.QObject | None:
        return self._quick_widget.rootObject()

    def _apply_or_buffer(self, op: Callable[[], None]) -> None:
        """Execute op immediately if QML is ready, otherwise buffer it."""
        if self._ready:
            op()
        else:
            self._pending_ops.append(op)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_center(self, lat: float, lon: float, zoom: int) -> None:
        """Position the map view at the given coordinates and zoom level.

        Buffers the operation if QML is not yet ready.
        """

        def _apply() -> None:
            root = self._root_object()
            if root:
                root.setProperty("initialLat", lat)
                root.setProperty("initialLon", lon)
                root.setProperty("initialZoom", zoom)

        self._apply_or_buffer(_apply)

    def set_marker(self, lat: float, lon: float) -> None:
        """Show a pin marker at the given coordinates.

        Buffers the operation if QML is not yet ready.
        """

        def _apply() -> None:
            root = self._root_object()
            if root:
                root.setProperty("markerLat", lat)
                root.setProperty("markerLon", lon)
                root.setProperty("showMarker", True)

        self._apply_or_buffer(_apply)

    def set_initial_bbox(self, bbox: GpsBoundingBox) -> None:
        """Fit the map to the given bounding box.

        Buffers the operation if QML is not yet ready.
        """

        def _apply() -> None:
            root = self._root_object()
            if root:
                _apply_initial_bbox(root, bbox)

        self._apply_or_buffer(_apply)

    def current_bbox(self) -> GpsBoundingBox | None:
        """Return the current bounding box from QML, or None if unavailable."""
        root = self._root_object()
        if not root:
            return None
        return _build_bbox(
            root.property("south"),
            root.property("west"),
            root.property("north"),
            root.property("east"),
        )

    def _on_zoom_in(self) -> None:
        root = self._root_object()
        if root:
            root.setProperty("pendingZoomDelta", 1)

    def _on_zoom_out(self) -> None:
        root = self._root_object()
        if root:
            root.setProperty("pendingZoomDelta", -1)
