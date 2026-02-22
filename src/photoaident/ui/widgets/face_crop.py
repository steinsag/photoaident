from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


class FaceCropWidget(QtWidgets.QWidget):
    """Display widget showing a face crop, photo thumbnail, and metadata.

    Pure display widget — receives all data from the caller via load().
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Face crop label (left side, fixed 300 px height)
        self.crop_label = QtWidgets.QLabel()
        self.crop_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.crop_label.setFixedSize(300, 300)
        self.crop_label.setStyleSheet("background-color: #aaa; border: 1px solid #888;")
        layout.addWidget(self.crop_label)

        # Right side: thumbnail + metadata
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)
        layout.addLayout(right)

        self.thumb_label = QtWidgets.QLabel()
        self.thumb_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setMaximumWidth(220)
        self.thumb_label.setStyleSheet(
            "background-color: #aaa; border: 1px solid #888;"
        )
        right.addWidget(self.thumb_label)

        self.meta_label = QtWidgets.QLabel()
        self.meta_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.meta_label.setWordWrap(True)
        right.addWidget(self.meta_label)

        right.addStretch()
        layout.addStretch()

        self.clear()

    def load(
        self,
        *,
        crop_path: Optional[Path],
        thumb_path: Optional[Path],
        taken_at: str,
        confidence: float,
    ) -> None:
        """Load and display the face crop, thumbnail, and metadata."""
        # Face crop
        if crop_path is not None and crop_path.exists():
            pix = QtGui.QPixmap(str(crop_path))
            if not pix.isNull():
                self.crop_label.setPixmap(
                    pix.scaledToHeight(
                        300, QtCore.Qt.TransformationMode.SmoothTransformation
                    )
                )
            else:
                self.crop_label.setPixmap(QtGui.QPixmap())
                self.crop_label.setText(self.tr("No image"))
        else:
            self.crop_label.setPixmap(QtGui.QPixmap())
            self.crop_label.setText(self.tr("No image"))

        # Photo thumbnail
        if thumb_path is not None and thumb_path.exists():
            pix = QtGui.QPixmap(str(thumb_path))
            if not pix.isNull():
                self.thumb_label.setPixmap(
                    pix.scaledToWidth(
                        220, QtCore.Qt.TransformationMode.SmoothTransformation
                    )
                )
            else:
                self.thumb_label.setPixmap(QtGui.QPixmap())
                self.thumb_label.setText(self.tr("No thumbnail"))
        else:
            self.thumb_label.setPixmap(QtGui.QPixmap())
            self.thumb_label.setText(self.tr("No thumbnail"))

        # Metadata
        self.meta_label.setText(
            self.tr("Date: {taken_at}\nConfidence: {confidence}%").format(
                taken_at=taken_at,
                confidence=f"{confidence * 100:.1f}",
            )
        )

    def clear(self) -> None:
        """Reset to an empty/placeholder state."""
        self.crop_label.setPixmap(QtGui.QPixmap())
        self.crop_label.setText("")
        self.thumb_label.setPixmap(QtGui.QPixmap())
        self.thumb_label.setText("")
        self.meta_label.setText("")
