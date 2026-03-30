import html
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.core.search_person import resolve_faces_to_persons
from photoaident.db.database import Face, FaceState, Image as DBImage
from photoaident.ui.widgets.map_widget import MapWidget
from photoaident.ui.window_state import restore_widget_geometry, save_widget_geometry
from photoaident.utils.file_manager import reveal_in_file_manager

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore
    from photoaident.paths import AppPaths


class _FaceOverlayLabel(QtWidgets.QLabel):
    """QLabel that shows a tooltip when the mouse hovers over a face bounding box."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._face_regions: list[tuple[QtCore.QRectF, str]] = []
        self._original_size: QtCore.QSize = QtCore.QSize()
        self._last_hovered_index: int = -1
        self.setMouseTracking(True)

    def set_face_regions(
        self,
        pixmap_regions: list[tuple[QtCore.QRectF, str]],
        pixmap_size: QtCore.QSize,
    ) -> None:
        """
        Set face bounding boxes and their tooltip text.
        Coordinates must be in the same coordinate space as the pixmap
        provided to setPixmap (e.g., after EXIF transformation but before
        UI-level scaling).
        """
        self._face_regions = pixmap_regions
        self._original_size = pixmap_size
        self._last_hovered_index = -2

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        if not self._face_regions or not self._original_size.isValid():
            return

        pm = self.pixmap()
        if pm is None or pm.isNull():
            return

        pm_w = pm.width()
        pm_h = pm.height()
        if pm_w == 0 or pm_h == 0:
            return

        # Center-alignment offsets (label may be larger than the pixmap)
        offset_x = (self.width() - pm_w) / 2
        offset_y = (self.height() - pm_h) / 2

        pos = event.position()
        # Map mouse position to pixmap coordinates
        px_x = (pos.x() - offset_x) / pm_w * self._original_size.width()
        px_y = (pos.y() - offset_y) / pm_h * self._original_size.height()
        current_hovered_index = -1

        for i, (rect, tooltip_text) in enumerate(self._face_regions):
            if rect.contains(px_x, px_y):
                current_hovered_index = i
                break

        if current_hovered_index != self._last_hovered_index:
            self._last_hovered_index = current_hovered_index
            if current_hovered_index != -1:
                _, tooltip_text = self._face_regions[current_hovered_index]
                QtWidgets.QToolTip.showText(
                    event.globalPosition().toPoint(), tooltip_text, self
                )
            else:
                QtWidgets.QToolTip.hideText()


class ImageDetailDialog(QtWidgets.QDialog):
    """
    A modal dialog that displays a full-size image with its metadata and
    face bounding boxes.
    """

    navigate_to_labelling = QtCore.Signal(int)  # image_id
    navigate_to_browse = QtCore.Signal(str)  # file_path

    def __init__(
        self,
        image: DBImage,
        session_factory: "sessionmaker",
        vector_store: "VectorStore",
        paths: "AppPaths",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Image Details"))
        self.setMinimumSize(800, 600)
        self.image_data = image
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._paths = paths
        self._resolved_names: dict[int, tuple[str, float] | None] = {}

        self._setup_ui()
        restore_widget_geometry(self, self._paths.window_state_file)
        self._load_image()

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in bytes to a human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(self._create_metadata_panel())
        main_layout.addWidget(self._create_image_area(), 1)

    def _create_metadata_panel(self) -> QtWidgets.QFrame:
        """Build and return the left metadata panel."""
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        panel.setFixedWidth(250)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        layout.addWidget(QtWidgets.QLabel("<b>" + self.tr("Metadata") + "</b>"))
        self._add_image_metadata_rows(layout)
        layout.addStretch()
        self._add_action_buttons(layout)

        return panel

    def _add_image_metadata_rows(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Populate layout with metadata rows and optional map widget."""
        self._add_meta_row(layout, self.tr("ID"), self.image_data.id)
        self._add_clickable_meta_row(
            layout, self.tr("File Path"), self.image_data.file_path
        )
        self._add_meta_row(
            layout,
            self.tr("File Size"),
            self._format_file_size(self.image_data.file_size),
        )

        if not self.image_data.metadata_rel:
            return

        meta = self.image_data.metadata_rel
        if meta.width and meta.height:
            self._add_meta_row(
                layout, self.tr("Dimensions"), f"{meta.width} x {meta.height}"
            )
        if meta.taken_at:
            self._add_meta_row(
                layout,
                self.tr("Taken At"),
                meta.taken_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
        if meta.camera_make or meta.camera_model:
            camera = f"{meta.camera_make or ''} {meta.camera_model or ''}".strip()
            self._add_meta_row(layout, self.tr("Camera"), camera)
        if meta.gps_lat is not None and meta.gps_lon is not None:
            map_widget = self._create_map_widget(
                float(meta.gps_lat), float(meta.gps_lon)
            )
            layout.addWidget(map_widget)

    def _add_meta_row(
        self,
        layout: QtWidgets.QVBoxLayout,
        label_text: str,
        value_text: str | int | None,
    ) -> None:
        """Add a two-column label/value row to *layout*. No-ops when value is None."""
        if value_text is None:
            return
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
        layout.addLayout(row_layout)

    def _add_clickable_meta_row(
        self,
        layout: QtWidgets.QVBoxLayout,
        label_text: str,
        value_text: str,
    ) -> None:
        """Add a two-column label/link row; clicking emits navigate_to_browse."""
        row_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(f"<b>{label_text}:</b>")
        label.setFixedWidth(80)
        escaped = html.escape(value_text)
        value = QtWidgets.QLabel(f'<a href="#">{escaped}</a>')
        value.setWordWrap(True)
        value.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
            | QtCore.Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        value.setOpenExternalLinks(False)
        value.linkActivated.connect(lambda _: self._on_browse_photo_folder_clicked())
        row_layout.addWidget(label)
        row_layout.addWidget(value)
        layout.addLayout(row_layout)

    def _create_map_widget(self, lat: float, lon: float) -> MapWidget:
        """Create a MapWidget centred on the given GPS coordinates."""
        map_widget = MapWidget(self._paths, show_overlay=False)
        map_widget.setFixedHeight(200)
        map_widget.set_center(lat, lon, zoom=14)
        map_widget.set_marker(lat, lon)
        return map_widget

    def _add_action_buttons(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Append action buttons to the layout."""
        browse_btn = QtWidgets.QPushButton(self.tr("Browse Photo Folder"))
        browse_btn.setAutoDefault(False)
        browse_btn.clicked.connect(self._on_browse_photo_folder_clicked)
        layout.addWidget(browse_btn)

        has_unidentified = any(
            f.state == FaceState.UNIDENTIFIED and f.deleted_at is None
            for f in self.image_data.faces
        )

        label_btn = QtWidgets.QPushButton(self.tr("Label Faces"))
        label_btn.setAutoDefault(False)
        label_btn.setEnabled(has_unidentified)
        label_btn.clicked.connect(self._on_label_faces_clicked)
        layout.addWidget(label_btn)

        show_btn = QtWidgets.QPushButton(self.tr("Show in File Manager"))
        show_btn.setAutoDefault(False)
        show_btn.clicked.connect(self._on_show_in_file_manager_clicked)
        layout.addWidget(show_btn)

        close_button = QtWidgets.QPushButton(self.tr("Close"))
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def _create_image_area(self) -> QtWidgets.QScrollArea:
        """Build and return the right-panel scroll area containing the image label."""
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.image_label = _FaceOverlayLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        return self.scroll_area

    def _get_face_display_info(self, face: Face) -> tuple[QtCore.Qt.GlobalColor, str]:
        """Return (color, tooltip) for a single face based on its state."""
        green = QtCore.Qt.GlobalColor.green
        red = QtCore.Qt.GlobalColor.red

        if face.state == FaceState.IDENTIFIED and face.person:
            return green, face.person.name

        if face.state == FaceState.ANONYMOUS:
            return green, self.tr("Anonymous")

        # UNIDENTIFIED (or IDENTIFIED without a person): check cached FAISS match
        match = self._resolved_names.get(face.id) if face.id is not None else None
        if match:
            name, score = match
            return green, f"{name} ({score:.0%})"

        return red, self.tr("Unknown")

    def _build_face_display_info(
        self,
    ) -> list[tuple[QtCore.QRectF, QtCore.Qt.GlobalColor, str]]:
        """Return (rect, color, tooltip) for every non-deleted face.

        Color is green for identified/anonymous/matched-unidentified faces and
        red only for truly unknown (unidentified with no FAISS match) faces.
        """
        # Pre-resolve unidentified faces in one batch to avoid per-face (N+1) queries
        unidentified_ids = [
            f.id
            for f in self.image_data.faces
            if f.state == FaceState.UNIDENTIFIED
            and f.deleted_at is None
            and f.id is not None
            and f.id not in self._resolved_names
        ]

        if unidentified_ids:
            resolved = resolve_faces_to_persons(
                unidentified_ids, self._session_factory, self._vector_store
            )
            self._resolved_names.update(resolved)

        result: list[tuple[QtCore.QRectF, QtCore.Qt.GlobalColor, str]] = []
        for face in self.image_data.faces:
            if face.deleted_at is not None:
                continue

            rect = QtCore.QRectF(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h)
            color, tooltip = self._get_face_display_info(face)
            result.append((rect, color, tooltip))

        return result

    def _load_image(self):
        file_path = Path(self.image_data.file_path)
        if not file_path.exists():
            self.image_label.setText(
                self.tr("Image file not found: {path}").format(path=file_path)
            )
            return

        # Load with EXIF orientation applied. Bounding boxes stored in the DB
        # are also in this EXIF-corrected coordinate space (process_image uses
        # open_image which calls ImageOps.exif_transpose before detection).
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

        # Build display info first so colors reflect FAISS match results
        face_display = self._build_face_display_info()

        if face_display:
            pen_width = max(2, pixmap.width() // 500)
            painter = QtGui.QPainter(pixmap)
            for rect, color, _ in face_display:
                pen = QtGui.QPen(color)
                pen.setWidth(pen_width)
                painter.setPen(pen)
                painter.drawRect(rect.toRect())
            painter.end()

        # Bboxes and pixmap are both in EXIF-corrected space — no post-transform needed.
        tooltip_regions = [(rect, tooltip) for rect, _, tooltip in face_display]

        self.image_label.set_face_regions(
            pixmap_regions=tooltip_regions, pixmap_size=pixmap.size()
        )

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

    def _on_browse_photo_folder_clicked(self) -> None:
        self.accept()
        self.navigate_to_browse.emit(str(self.image_data.file_path))

    def _on_label_faces_clicked(self) -> None:
        self.accept()
        self.navigate_to_labelling.emit(self.image_data.id)

    def _on_show_in_file_manager_clicked(self) -> None:
        reveal_in_file_manager(self.image_data.file_path)

    def done(self, result: int) -> None:
        """Save geometry before closing."""
        save_widget_geometry(self, self._paths.window_state_file)
        super().done(result)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Re-scale image when dialog is resized
        QtCore.QTimer.singleShot(10, self._update_image_display)
