from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


class FaceCropWidget(QtWidgets.QLabel):
    """Displays a face crop thumbnail at a fixed 300×300 size."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setFixedSize(300, 300)
        self.setStyleSheet("background-color: #aaa; border: 1px solid #888;")

    def load(self, crop_path: Optional[Path]) -> None:
        """Show a scaled crop pixmap, or placeholder text if unavailable."""
        if crop_path is not None and crop_path.exists():
            pix = QtGui.QPixmap(str(crop_path))
            if not pix.isNull():
                self.setText("")
                self.setPixmap(
                    pix.scaledToHeight(
                        300, QtCore.Qt.TransformationMode.SmoothTransformation
                    )
                )
                return
        self.setPixmap(QtGui.QPixmap())
        self.setText(self.tr("No image"))

    def clear(self) -> None:
        """Reset to empty state."""
        self.setPixmap(QtGui.QPixmap())
        self.setText("")
