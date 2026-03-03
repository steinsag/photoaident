import sys
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.core.geo import GpsBoundingBox


def _icon_path(name: str) -> str:
    """Return the path to an icon, works for dev and PyInstaller bundles."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return str(Path(meipass) / "assets" / "icons" / name)
    # map_preview.py lives at src/photoaident/ui/widgets/ — go up 5 levels
    return str(
        Path(__file__).parent.parent.parent.parent.parent / "assets" / "icons" / name
    )


class MapPreviewWidget(QtWidgets.QFrame):
    """Small widget for the filter panel showing the selected GPS location.

    Shows either a clickable text "Click to set location" or the selected
    bounding box coordinates. Includes a "Clear" button when a location is set.
    """

    clicked = QtCore.Signal()
    cleared = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)

        self._bbox: Optional[GpsBoundingBox] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QtWidgets.QLabel(self.tr("Location"))
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        # Main area (clickable)
        self._content_btn = QtWidgets.QPushButton()
        self._content_btn.setFlat(True)
        self._content_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._content_btn.setIcon(QtGui.QIcon(_icon_path("world_map.svg")))
        self._content_btn.setIconSize(QtCore.QSize(24, 24))
        self._content_btn.clicked.connect(self.clicked.emit)
        layout.addWidget(self._content_btn)

        # Clear button
        self._clear_btn = QtWidgets.QPushButton(self.tr("Clear Location"))
        self._clear_btn.clicked.connect(self.cleared.emit)
        layout.addWidget(self._clear_btn)

        self._update_display()

    def set_bbox(self, bbox: Optional[GpsBoundingBox]) -> None:
        """Set the current bounding box and update the UI."""
        self._bbox = bbox
        self._update_display()

    def _update_display(self) -> None:
        """Update buttons based on whether a bbox is set."""
        if self._bbox is None:
            self._content_btn.setText(self.tr("Click to set location"))
            self._clear_btn.hide()
        else:
            # Display center coordinate and span as a summary
            lat = (self._bbox.south + self._bbox.north) / 2
            lon = (self._bbox.west + self._bbox.east) / 2
            text = f"Lat: {lat:.4f}, Lon: {lon:.4f}\n"
            lat_span = self._bbox.north - self._bbox.south
            lon_span = abs(self._bbox.east - self._bbox.west)
            text += f"(Span: {lat_span:.2f}° x {lon_span:.2f}°)"
            self._content_btn.setText(text)
            self._clear_btn.show()
