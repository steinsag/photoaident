from PySide6 import QtWidgets

from photoaident.core.geo import GpsBoundingBox
from photoaident.paths import AppPaths
from photoaident.ui.widgets.map_widget import MapWidget
from photoaident.ui.window_state import restore_widget_geometry, save_widget_geometry


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

        instruction = QtWidgets.QLabel(
            self.tr(
                "Pan and zoom the map. The highlighted area defines the search region."
            )
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        self._map_widget = MapWidget(self._paths, show_overlay=True)
        layout.addWidget(self._map_widget, stretch=1)

        if initial_bbox is not None:
            self._map_widget.set_initial_bbox(initial_bbox)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def done(self, result: int) -> None:
        """Save geometry before closing."""
        save_widget_geometry(self, self._paths.window_state_file)
        super().done(result)

    def _on_accept(self) -> None:
        self._selected_bbox = self._map_widget.current_bbox()
        self.accept()

    def selected_bbox(self) -> GpsBoundingBox | None:
        """Return the selected GpsBoundingBox, or None if canceled."""
        return self._selected_bbox
