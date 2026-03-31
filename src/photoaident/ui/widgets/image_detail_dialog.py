import html
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from photoaident.core.search import SearchResult
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
        current_index: int = 0,
        all_results: list[SearchResult] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Image Details"))
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.image_data = image
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._paths = paths
        self._resolved_names: dict[int, tuple[str, float] | None] = {}
        self._current_index = current_index
        self._all_results = all_results or []
        self._metadata_widgets: dict[str, QtWidgets.QLabel] = {}
        self._metadata_row_containers: dict[str, QtWidgets.QWidget] = {}
        self._label_btn: QtWidgets.QPushButton | None = None
        self._map_widget: MapWidget | None = None
        self._metadata_layout: QtWidgets.QVBoxLayout | None = None
        self._map_container: QtWidgets.QWidget | None = None

        self._setup_ui()
        QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self
        ).activated.connect(self._show_previous_image)
        QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self
        ).activated.connect(self._show_next_image)
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
        main_layout.addWidget(self._create_image_container(), 1)

    def _create_image_container(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._create_image_area(), 1)

        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addStretch()

        self._prev_btn = QtWidgets.QPushButton("< " + self.tr("Previous"))
        self._prev_btn.setAutoDefault(False)
        self._prev_btn.clicked.connect(self._show_previous_image)
        nav_layout.addWidget(self._prev_btn)

        self._nav_label = QtWidgets.QLabel()
        self._nav_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._nav_label)

        self._next_btn = QtWidgets.QPushButton(self.tr("Next") + " >")
        self._next_btn.setAutoDefault(False)
        self._next_btn.clicked.connect(self._show_next_image)
        nav_layout.addWidget(self._next_btn)

        nav_layout.addStretch()

        layout.addLayout(nav_layout)
        self._update_navigation()
        return container

    def _create_metadata_panel(self) -> QtWidgets.QFrame:
        """Build and return the left metadata panel."""
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        panel.setFixedWidth(250)
        self._metadata_layout = QtWidgets.QVBoxLayout(panel)
        self._metadata_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self._metadata_layout.addWidget(
            QtWidgets.QLabel("<b>" + self.tr("Metadata") + "</b>")
        )
        self._map_container = QtWidgets.QWidget()
        self._map_container.setLayout(QtWidgets.QVBoxLayout())
        self._add_image_metadata_rows(self._metadata_layout)
        self._metadata_layout.addWidget(self._map_container)
        self._metadata_layout.addStretch()
        self._add_action_buttons(self._metadata_layout)

        return panel

    def _add_image_metadata_rows(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Populate layout with always-present rows and pre-created optional rows."""
        self._add_meta_row(layout, self.tr("ID"), self.image_data.id, "id")
        self._add_clickable_meta_row(
            layout, self.tr("File Path"), self.image_data.file_path, "file_path"
        )
        self._add_meta_row(
            layout,
            self.tr("File Size"),
            self._format_file_size(self.image_data.file_size),
            "file_size",
        )
        self._pre_create_optional_row(layout, self.tr("Dimensions"), "dimensions")
        self._pre_create_optional_row(layout, self.tr("Taken At"), "taken_at")
        self._pre_create_optional_row(layout, self.tr("Camera"), "camera")

        meta = self.image_data.metadata_rel
        self._update_optional_field(
            present=bool(meta and meta.width and meta.height),
            value=(
                f"{meta.width} x {meta.height}"
                if meta and meta.width and meta.height
                else ""
            ),
            key="dimensions",
        )
        self._update_optional_field(
            present=bool(meta and meta.taken_at),
            value=(
                meta.taken_at.strftime("%Y-%m-%d %H:%M:%S")
                if meta and meta.taken_at
                else ""
            ),
            key="taken_at",
        )
        camera = (
            f"{meta.camera_make or ''} {meta.camera_model or ''}".strip()
            if meta and (meta.camera_make or meta.camera_model)
            else ""
        )
        self._update_optional_field(present=bool(camera), value=camera, key="camera")
        if meta and meta.gps_lat is not None and meta.gps_lon is not None:
            self._create_map_widget(float(meta.gps_lat), float(meta.gps_lon))
            map_layout = (
                self._map_container.layout()
                if self._map_container is not None
                else None
            )
            if map_layout is not None and self._map_widget is not None:
                map_layout.addWidget(self._map_widget)

    def _add_meta_row(
        self,
        layout: QtWidgets.QVBoxLayout,
        label_text: str,
        value_text: str | int | None,
        key: str | None = None,
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
        if key is not None:
            self._metadata_widgets[key] = value

    def _add_clickable_meta_row(
        self,
        layout: QtWidgets.QVBoxLayout,
        label_text: str,
        value_text: str,
        key: str | None = None,
    ) -> None:
        """Add a two-column label/link row; clicking emits navigate_to_browse."""
        row_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(f"<b>{label_text}:</b>")
        label.setFixedWidth(80)
        escaped = html.escape(value_text)
        value = QtWidgets.QLabel(f'<a href="#">{escaped}</a>')
        value.setWordWrap(True)
        value.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
            | QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
            | QtCore.Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        value.setOpenExternalLinks(False)
        value.linkActivated.connect(lambda _: self._on_browse_photo_folder_clicked())
        row_layout.addWidget(label)
        row_layout.addWidget(value)
        layout.addLayout(row_layout)
        if key is not None:
            self._metadata_widgets[key] = value

    def _pre_create_optional_row(
        self,
        layout: QtWidgets.QVBoxLayout,
        label_text: str,
        key: str,
    ) -> None:
        """Create a hidden container widget for an optional metadata row."""
        container = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(f"<b>{label_text}:</b>")
        label.setFixedWidth(80)
        value = QtWidgets.QLabel()
        value.setWordWrap(True)
        value.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row_layout.addWidget(label)
        row_layout.addWidget(value)
        layout.addWidget(container)
        container.setVisible(False)
        self._metadata_widgets[key] = value
        self._metadata_row_containers[key] = container

    def _create_map_widget(self, lat: float, lon: float) -> MapWidget:
        """Create a MapWidget centred on the given GPS coordinates."""
        map_widget = MapWidget(self._paths, show_overlay=False)
        map_widget.setFixedHeight(200)
        map_widget.set_center(lat, lon, zoom=14)
        map_widget.set_marker(lat, lon)
        self._map_widget = map_widget
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

        self._label_btn = QtWidgets.QPushButton(self.tr("Label Faces"))
        self._label_btn.setAutoDefault(False)
        self._label_btn.setEnabled(has_unidentified)
        self._label_btn.clicked.connect(self._on_label_faces_clicked)
        layout.addWidget(self._label_btn)

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

    def _update_navigation(self) -> None:
        total = len(self._all_results)
        has_nav = total > 0
        self._prev_btn.setEnabled(has_nav and self._current_index > 0)
        self._next_btn.setEnabled(has_nav and self._current_index < total - 1)
        if total > 0:
            self._nav_label.setText(f"{self._current_index + 1} / {total}")
        else:
            self._nav_label.setText("")

    def _show_previous_image(self) -> None:
        if self._current_index > 0:
            self._current_index -= 1
            self._load_image_by_index()

    def _show_next_image(self) -> None:
        if self._current_index < len(self._all_results) - 1:
            self._current_index += 1
            self._load_image_by_index()

    def _load_image_by_index(self) -> None:
        result = self._all_results[self._current_index]
        with self._session_factory() as session:
            stmt = (
                select(DBImage)
                .where(DBImage.id == result.image_id)
                .options(
                    joinedload(DBImage.faces).joinedload(Face.person),
                    joinedload(DBImage.metadata_rel),
                )
            )
            image = session.execute(stmt).unique().scalar_one_or_none()
            if image:
                self.image_data = image
                self._resolved_names.clear()
                self._load_image()
                self._update_metadata_panel()
                self._update_action_buttons()
        self._update_navigation()

    def _update_action_buttons(self) -> None:
        """Update action button states to reflect the current image."""
        if self._label_btn is None:
            return
        has_unidentified = any(
            f.state == FaceState.UNIDENTIFIED and f.deleted_at is None
            for f in self.image_data.faces
        )
        self._label_btn.setEnabled(has_unidentified)

    def _update_metadata_panel(self) -> None:
        """Update the metadata panel widgets with data from the current image."""
        image = self.image_data

        self._metadata_widgets["id"].setText(str(image.id))
        self._metadata_widgets["file_path"].setText(
            f'<a href="#">{html.escape(image.file_path)}</a>'
        )
        self._metadata_widgets["file_size"].setText(
            self._format_file_size(image.file_size)
        )

        meta = image.metadata_rel

        self._update_optional_field(
            present=bool(meta and meta.width and meta.height),
            value=(
                f"{meta.width} x {meta.height}"
                if meta and meta.width and meta.height
                else ""
            ),
            key="dimensions",
        )

        self._update_optional_field(
            present=bool(meta and meta.taken_at is not None),
            value=(
                meta.taken_at.strftime("%Y-%m-%d %H:%M:%S")
                if meta and meta.taken_at
                else ""
            ),
            key="taken_at",
        )

        camera = (
            f"{meta.camera_make or ''} {meta.camera_model or ''}".strip()
            if meta and (meta.camera_make or meta.camera_model)
            else ""
        )
        self._update_optional_field(
            present=bool(camera),
            value=camera,
            key="camera",
        )

        self._update_map_widget(
            present=bool(
                meta and meta.gps_lat is not None and meta.gps_lon is not None
            ),
            lat=float(meta.gps_lat) if meta and meta.gps_lat else 0.0,
            lon=float(meta.gps_lon) if meta and meta.gps_lon else 0.0,
        )

    def _update_optional_field(
        self,
        present: bool,
        value: str,
        key: str,
    ) -> None:
        """Show or hide a pre-created optional metadata row."""
        container = self._metadata_row_containers.get(key)
        widget = self._metadata_widgets.get(key)
        if container is not None:
            container.setVisible(present)
        if widget is not None and present:
            widget.setText(value)

    def _update_map_widget(self, present: bool, lat: float, lon: float) -> None:
        """Show or hide the map widget, creating it if needed."""
        if self._map_container is None:
            return
        if not present:
            self._map_container.setVisible(False)
            return
        if self._map_widget is None:
            map_layout = self._map_container.layout()
            if map_layout is not None:
                self._map_widget = self._create_map_widget(lat, lon)
                map_layout.addWidget(self._map_widget)
        else:
            self._map_widget.set_center(lat, lon, zoom=14)
            self._map_widget.set_marker(lat, lon)
        self._map_container.setVisible(True)

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
