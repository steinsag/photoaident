from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.db.database import Image as DBImage


class ImageDetailDialog(QtWidgets.QDialog):
    """
    A modal dialog that displays a full-size image with its metadata and
    face bounding boxes.
    """

    def __init__(self, image: DBImage, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Image Details"))
        self.setMinimumSize(800, 600)
        self.image_data = image

        self._setup_ui()
        self._load_image()

    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left panel: Metadata
        metadata_panel = QtWidgets.QFrame()
        metadata_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        metadata_panel.setFixedWidth(250)
        metadata_layout = QtWidgets.QVBoxLayout(metadata_panel)
        metadata_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        title_label = QtWidgets.QLabel(f"<b>{self.tr('Metadata')}</b>")
        metadata_layout.addWidget(title_label)

        # Helper to add metadata rows
        def add_meta(label_text, value_text):
            row_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(f"<b>{label_text}:</b>")
            label.setFixedWidth(80)
            value = QtWidgets.QLabel(str(value_text))
            value.setWordWrap(True)
            value.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            row_layout.addWidget(label)
            row_layout.addWidget(value)
            metadata_layout.addLayout(row_layout)

        add_meta(self.tr("ID"), self.image_data.id)
        add_meta(self.tr("File Path"), self.image_data.file_path)

        # Format file size
        size_bytes = self.image_data.file_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        add_meta(self.tr("File Size"), size_str)

        if self.image_data.metadata_rel:
            meta = self.image_data.metadata_rel
            if meta.width and meta.height:
                add_meta(self.tr("Dimensions"), f"{meta.width} x {meta.height}")
            if meta.taken_at:
                add_meta(
                    self.tr("Taken At"), meta.taken_at.strftime("%Y-%m-%d %H:%M:%S")
                )
            if meta.camera_make or meta.camera_model:
                camera = f"{meta.camera_make or ''} {meta.camera_model or ''}".strip()
                add_meta(self.tr("Camera"), camera)

        metadata_layout.addStretch()

        close_button = QtWidgets.QPushButton(self.tr("Close"))
        close_button.clicked.connect(self.accept)
        metadata_layout.addWidget(close_button)

        main_layout.addWidget(metadata_panel)

        # Right panel: Image View
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        main_layout.addWidget(self.scroll_area, 1)

    def _load_image(self):
        file_path = Path(self.image_data.file_path)
        if not file_path.exists():
            self.image_label.setText(
                self.tr("Image file not found: {path}").format(path=file_path)
            )
            return

        # Load image with QImageReader to respect EXIF orientation
        reader = QtGui.QImageReader(str(file_path))
        reader.setAutoTransform(True)
        qimage = reader.read()

        if qimage.isNull():
            self.image_label.setText(
                self.tr("Failed to load image: {error}").format(
                    error=reader.errorString()
                )
            )
            return

        pixmap = QtGui.QPixmap.fromImage(qimage)

        # Draw bounding boxes
        if self.image_data.faces:
            painter = QtGui.QPainter(pixmap)
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red)
            pen.setWidth(
                max(2, pixmap.width() // 500)
            )  # Scale pen width with image size
            painter.setPen(pen)

            for face in self.image_data.faces:
                rect = QtCore.QRect(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h)
                painter.drawRect(rect)

            painter.end()

        # Scale pixmap to fit the view initially, but allow scrolling
        # Actually, let's just show it scaled to fit the scroll area by default
        self._original_pixmap = pixmap
        self._update_image_display()

    def _update_image_display(self):
        if not hasattr(self, "_original_pixmap"):
            return

        # Scaling logic to fit nicely in the dialog but keep it readable
        available_size = self.scroll_area.viewport().size()
        scaled_pixmap = self._original_pixmap.scaled(
            available_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Re-scale image when dialog is resized
        QtCore.QTimer.singleShot(10, self._update_image_display)
