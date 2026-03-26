import logging
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets, QtQuickWidgets

from photoaident.core.geo import GpsBoundingBox
from photoaident.paths import AppPaths
from photoaident.ui.window_state import restore_widget_geometry, save_widget_geometry
from photoaident.utils.resource_path import icon_path as _icon_path

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
        initial_bbox: GpsBoundingBox | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Select Location"))
        self.resize(1024, 768)
        restore_widget_geometry(self, paths.window_state_file)

        self._selected_bbox: GpsBoundingBox | None = None
        self._paths = paths

        self._setup_ui(initial_bbox)

    def _setup_ui(self, initial_bbox: GpsBoundingBox | None = None) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._setup_instruction_label(layout)
        self._setup_map_widget(layout, initial_bbox)
        self._setup_zoom_buttons(layout)
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
        self, layout: QtWidgets.QVBoxLayout, initial_bbox: GpsBoundingBox | None
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

        # Store for use by the status handler (may fire after setSource returns).
        self._pending_initial_bbox = initial_bbox

        # Create the tiles directory before QML initialises the OSM plugin so
        # the network disk-cache path is valid from the very first tile fetch.
        self._paths.tiles_dir.mkdir(parents=True, exist_ok=True)

        # Inject cachePath before setSource so the Plugin sees a valid directory
        # during its one-time initialisation (PluginParameter bindings are not
        # reactive after the plugin is already resolved).
        self._quick_widget.setInitialProperties(
            {"cachePath": str(self._paths.tiles_dir)}
        )

        # Connect before setSource so we catch both synchronous and asynchronous
        # status transitions (Loading → Ready / Error).
        self._quick_widget.statusChanged.connect(self._on_qml_status_changed)

        qml_path = Path(__file__).parent / "map_view.qml"
        self._quick_widget.setSource(QtCore.QUrl.fromLocalFile(str(qml_path)))

        layout.addWidget(self._quick_widget, stretch=1)

    def _on_qml_status_changed(
        self, status: QtQuickWidgets.QQuickWidget.Status
    ) -> None:
        """Handle QML load completion: apply initial state or log errors."""
        if status == QtQuickWidgets.QQuickWidget.Status.Error:
            for err in self._quick_widget.errors():
                logger.error("QML error: %s", err)
        elif status == QtQuickWidgets.QQuickWidget.Status.Ready:
            root_obj = self._quick_widget.rootObject()
            if root_obj and self._pending_initial_bbox:
                self._apply_initial_bbox(root_obj, self._pending_initial_bbox)

    @staticmethod
    def _apply_initial_bbox(root_obj: object, initial_bbox: GpsBoundingBox) -> None:
        """Seed the QML map with an initial bounding box.

        Two things happen here:

        1. The extraction properties (``south``/``west``/``north``/``east``) are
           pre-populated directly so that ``_extract_bbox`` always returns the
           correct value even when the user clicks OK before the map widget has
           been laid out and ``updateBbox()`` has had a chance to run.

        2. The pending-bbox properties are set and ``pendingBbox`` is flipped to
           ``true``.  QML's ``onPendingBboxChanged`` handler picks this up and
           calls ``_applyPendingBboxIfReady``, which positions the map view to
           show the bbox (deferred until the widget has non-zero dimensions).

        All communication uses ``setProperty`` — a reliable C++ API — instead of
        calling QML JavaScript functions across the Python/QML boundary, which can
        fail silently when PySide6 cannot resolve the method signature.
        """
        for name, value in (
            ("south", initial_bbox.south),
            ("west", initial_bbox.west),
            ("north", initial_bbox.north),
            ("east", initial_bbox.east),
            ("pendingBboxSouth", initial_bbox.south),
            ("pendingBboxWest", initial_bbox.west),
            ("pendingBboxNorth", initial_bbox.north),
            ("pendingBboxEast", initial_bbox.east),
        ):
            root_obj.setProperty(name, value)  # type: ignore[attr-defined]
        # Setting pendingBbox last triggers onPendingBboxChanged in QML.
        root_obj.setProperty("pendingBbox", True)  # type: ignore[attr-defined]

    def _setup_zoom_buttons(self, layout: QtWidgets.QVBoxLayout) -> None:
        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addStretch()

        self._zoom_out_btn = QtWidgets.QPushButton(self.tr("Zoom out"))
        self._zoom_out_btn.setIcon(QtGui.QIcon(_icon_path("zoom-out.svg")))
        self._zoom_out_btn.clicked.connect(self._on_zoom_out)
        zoom_layout.addWidget(self._zoom_out_btn)

        self._zoom_in_btn = QtWidgets.QPushButton(self.tr("Zoom in"))
        self._zoom_in_btn.setIcon(QtGui.QIcon(_icon_path("zoom-in.svg")))
        self._zoom_in_btn.clicked.connect(self._on_zoom_in)
        zoom_layout.addWidget(self._zoom_in_btn)

        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)

    def _on_zoom_in(self) -> None:
        root_obj = self._quick_widget.rootObject()
        if root_obj:
            # Trigger zoom via a QML property instead of calling JS functions directly
            root_obj.setProperty("pendingZoomDelta", 1)

    def _on_zoom_out(self) -> None:
        root_obj = self._quick_widget.rootObject()
        if root_obj:
            # Trigger zoom via a QML property instead of calling JS functions directly
            root_obj.setProperty("pendingZoomDelta", -1)

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
    ) -> GpsBoundingBox | None:
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

    def _extract_bbox(self) -> GpsBoundingBox | None:
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

    def done(self, result: int) -> None:
        """Save geometry before closing."""
        save_widget_geometry(self, self._paths.window_state_file)
        super().done(result)

    def _on_accept(self) -> None:
        self._selected_bbox = self._extract_bbox()
        self.accept()

    def selected_bbox(self) -> GpsBoundingBox | None:
        """Return the selected GpsBoundingBox, or None if canceled."""
        return self._selected_bbox
