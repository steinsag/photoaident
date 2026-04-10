from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.utils.image_utils import extract_and_save_face_crop


class FaceCropWidget(QtWidgets.QLabel):
    """Displays a face crop thumbnail at a fixed 300×300 size."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setFixedSize(300, 300)
        self.setStyleSheet("background-color: #aaa; border: 1px solid #888;")

    def load(
        self,
        crop_path: Path | None,
        image_path: Path | None = None,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Show a scaled crop pixmap, regenerating it if missing.

        If crop_path does not exist but image_path and bbox are provided,
        the crop is regenerated from the source image and saved to crop_path.
        Falls back to a placeholder if the crop cannot be loaded or created.

        Args:
            crop_path: Path to the cached face crop JPEG.
            image_path: Path to the original source image (for regeneration).
            bbox: (x, y, w, h) bounding box as stored in the database.
        """
        self.setPixmap(QtGui.QPixmap())
        self.setText("")

        if crop_path is None:
            self.setText(self.tr("No image"))
            return

        if not crop_path.exists() and image_path is not None and bbox is not None:
            if image_path.exists():
                extract_and_save_face_crop(image_path, bbox, crop_path)

        if crop_path.exists():
            pix = QtGui.QPixmap(str(crop_path))
            if pix.isNull() and image_path is not None and bbox is not None:
                # Corrupt or unreadable cache file — only remove and regenerate
                # when the source image is available; otherwise keep the file so
                # it isn't lost permanently.
                if image_path.exists():
                    crop_path.unlink(missing_ok=True)
                    extract_and_save_face_crop(image_path, bbox, crop_path)
                    pix = QtGui.QPixmap(str(crop_path)) if crop_path.exists() else pix
            if not pix.isNull():
                self.setPixmap(
                    pix.scaled(
                        300,
                        300,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                )
                return

        self.setText(self.tr("No image"))

    def clear(self) -> None:
        """Reset to empty state."""
        self.setPixmap(QtGui.QPixmap())
        self.setText("")
