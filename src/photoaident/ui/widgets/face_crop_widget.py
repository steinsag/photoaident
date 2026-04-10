from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.utils.image_utils import ensure_face_crop

_Bbox = tuple[int, int, int, int]


class FaceCropWidget(QtWidgets.QLabel):
    """Displays a face crop thumbnail at a fixed square size."""

    def __init__(
        self, size: int = 300, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._size = size
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setFixedSize(size, size)
        self.setStyleSheet("background-color: #aaa; border: 1px solid #888;")

    def load(
        self,
        crop_path: Path | None,
        image_path: Path | None = None,
        bbox: _Bbox | None = None,
    ) -> None:
        """Show a scaled crop pixmap, regenerating it if missing or corrupt.

        Delegates all file-level operations (missing/corrupt detection and
        regeneration) to ``ensure_face_crop`` in image_utils.  Falls back to
        a placeholder label if no valid crop can be produced.

        Args:
            crop_path: Path to the cached face crop JPEG.
            image_path: Path to the original source image (for regeneration).
            bbox: (x, y, w, h) bounding box as stored in the database.
        """
        self.setPixmap(QtGui.QPixmap())
        self.setText("")

        if crop_path is None or not ensure_face_crop(crop_path, image_path, bbox):
            self.setText(self.tr("No image"))
            return

        pixmap = QtGui.QPixmap(str(crop_path))
        if pixmap.isNull():
            self.setText(self.tr("No image"))
            return

        self.setPixmap(
            pixmap.scaled(
                self._size,
                self._size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def clear(self) -> None:
        """Reset to empty state."""
        self.setPixmap(QtGui.QPixmap())
        self.setText("")
