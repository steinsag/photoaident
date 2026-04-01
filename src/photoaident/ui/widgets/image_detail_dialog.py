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
    """A modal dialog that displays a full-size image with metadata and face boxes.

    Accepts a list of search results and a starting index so the user can
    navigate between images with Previous / Next buttons or arrow keys.
    """

    navigate_to_labelling = QtCore.Signal(int)  # image_id
    navigate_to_browse = QtCore.Signal(str)  # file_path

    _RESIZE_DEBOUNCE_MS = 50

    def __init__(
        self,
        results: list[SearchResult],
        current_index: int,
        session_factory: "sessionmaker",
        vector_store: "VectorStore",
        paths: "AppPaths",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Image Details"))
        self.setMinimumSize(800, 600)

        self._results = results
        self._current_index = current_index
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._paths = paths

        self._image_data: DBImage | None = None
        self._original_pixmap: QtGui.QPixmap | None = None
        self._resolved_names: dict[int, tuple[str, float] | None] = {}

        # Created here (before the dialog's native window exists) so that
        # QQuickWidget's OpenGL initialisation is included in the initial
        # window creation. If it were created later — while the dialog is
        # already visible — Qt would have to recreate the native window
        # handle, causing the dialog to briefly disappear.
        self._map_widget = MapWidget(self._paths, show_overlay=False)
        self._map_widget.setFixedHeight(200)
        self._map_widget.hide()

        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(self._RESIZE_DEBOUNCE_MS)
        self._resize_timer.timeout.connect(self._update_image_display)

        self._setup_ui()
        self._setup_shortcuts()
        restore_widget_geometry(self, self._paths.window_state_file)
        self._show_current_image()

    # ------------------------------------------------------------------
    # DB loading
    # ------------------------------------------------------------------

    def _load_image_data(self, image_id: int) -> DBImage | None:
        """Load a fully-populated Image from the DB (faces, persons, metadata)."""
        with self._session_factory() as session:
            stmt = (
                select(DBImage)
                .where(DBImage.id == image_id)
                .options(
                    joinedload(DBImage.faces).joinedload(Face.person),
                    joinedload(DBImage.metadata_rel),
                )
            )
            return session.execute(stmt).unique().scalar_one_or_none()

    # ------------------------------------------------------------------
    # UI setup (called once)
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(self._create_metadata_panel())
        main_layout.addWidget(self._create_image_area(), 1)

    def _create_metadata_panel(self) -> QtWidgets.QFrame:
        """Build and return the left metadata panel with a dynamic content area."""
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        panel.setFixedWidth(250)

        layout = QtWidgets.QVBoxLayout(panel)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        layout.addWidget(QtWidgets.QLabel("<b>" + self.tr("Metadata") + "</b>"))

        # Dynamic container — cleared and rebuilt on each image switch
        self._metadata_container = QtWidgets.QWidget()
        self._metadata_layout = QtWidgets.QVBoxLayout(self._metadata_container)
        self._metadata_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._metadata_container)
        layout.addWidget(self._map_widget)
        layout.addStretch()
        self._add_action_buttons(layout)

        return panel

    def _rebuild_metadata_content(self) -> None:
        """Clear and re-populate the dynamic metadata rows and optional map."""
        self._map_widget.hide()
        self._clear_layout(self._metadata_layout)

        if self._image_data is None:
            return

        self._add_image_metadata_rows(self._metadata_layout)

    @staticmethod
    def _clear_layout(layout: QtWidgets.QLayout) -> None:
        """Recursively remove and delete all items from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget() if item else None
            child_layout = item.layout() if item else None
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                ImageDetailDialog._clear_layout(child_layout)

    def _add_image_metadata_rows(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Populate layout with metadata rows and optional map widget."""
        assert self._image_data is not None

        self._add_meta_row(layout, self.tr("ID"), self._image_data.id)
        self._add_clickable_meta_row(
            layout, self.tr("File Path"), self._image_data.file_path
        )
        self._add_meta_row(
            layout,
            self.tr("File Size"),
            self._format_file_size(self._image_data.file_size),
        )

        if not self._image_data.metadata_rel:
            return

        meta = self._image_data.metadata_rel
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
            lat, lon = float(meta.gps_lat), float(meta.gps_lon)
            self._map_widget.set_center(lat, lon, zoom=14)
            self._map_widget.set_marker(lat, lon)
            self._map_widget.show()

    def _add_meta_row(
        self,
        layout: QtWidgets.QVBoxLayout,
        label_text: str,
        value_text: str | int | None,
    ) -> None:
        """Add a two-column label/value row to *layout*. No-ops when value is None."""
        if value_text is None:
            return
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(f"<b>{label_text}:</b>")
        label.setFixedWidth(80)
        value = QtWidgets.QLabel(str(value_text))
        value.setWordWrap(True)
        value.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row_layout.addWidget(label)
        row_layout.addWidget(value)
        layout.addWidget(row)

    def _add_clickable_meta_row(
        self,
        layout: QtWidgets.QVBoxLayout,
        label_text: str,
        value_text: str,
    ) -> None:
        """Add a two-column label/link row; clicking emits navigate_to_browse."""
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
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
        layout.addWidget(row)

    def _add_action_buttons(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Append action buttons to the layout."""
        browse_btn = QtWidgets.QPushButton(self.tr("Browse Photo Folder"))
        browse_btn.setAutoDefault(False)
        browse_btn.clicked.connect(self._on_browse_photo_folder_clicked)
        layout.addWidget(browse_btn)

        self._label_btn = QtWidgets.QPushButton(self.tr("Label Faces"))
        self._label_btn.setAutoDefault(False)
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

    def _create_image_area(self) -> QtWidgets.QWidget:
        """Build the right panel: scroll area for the image + navigation buttons."""
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.image_label = _FaceOverlayLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        container_layout.addWidget(self.scroll_area, 1)

        # Navigation buttons
        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addStretch()

        self._prev_btn = QtWidgets.QPushButton(self.tr("Previous"))
        self._prev_btn.setAutoDefault(False)
        self._prev_btn.clicked.connect(self._navigate_previous)
        nav_layout.addWidget(self._prev_btn)

        self._next_btn = QtWidgets.QPushButton(self.tr("Next"))
        self._next_btn.setAutoDefault(False)
        self._next_btn.clicked.connect(self._navigate_next)
        nav_layout.addWidget(self._next_btn)

        nav_layout.addStretch()
        container_layout.addLayout(nav_layout)

        return container

    # ------------------------------------------------------------------
    # Image switching (core of the refactoring)
    # ------------------------------------------------------------------

    def _show_current_image(self) -> None:
        """Load and display the image at _current_index."""
        result = self._results[self._current_index]
        self._image_data = self._load_image_data(result.image_id)
        self._resolved_names.clear()
        self._update_navigation_state()

        if self._image_data is None:
            self.image_label.setText(
                self.tr("Image not found in database (ID {id})").format(
                    id=result.image_id
                )
            )
            self._rebuild_metadata_content()
            return

        self._rebuild_metadata_content()
        self._update_label_button()
        self._load_image()

    def _update_navigation_state(self) -> None:
        """Enable/disable Previous and Next buttons based on position in list."""
        self._prev_btn.setEnabled(self._current_index > 0)
        self._next_btn.setEnabled(self._current_index < len(self._results) - 1)

    def _update_label_button(self) -> None:
        """Enable the Label Faces button only when there are unidentified faces."""
        if self._image_data is None:
            self._label_btn.setEnabled(False)
            return
        has_unidentified = any(
            f.state == FaceState.UNIDENTIFIED and f.deleted_at is None
            for f in self._image_data.faces
        )
        self._label_btn.setEnabled(has_unidentified)

    def _setup_shortcuts(self) -> None:
        """Register Left/Right arrow key shortcuts for image navigation."""
        self._shortcut_prev = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self
        )
        self._shortcut_prev.activated.connect(self._navigate_previous)
        self._shortcut_next = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self
        )
        self._shortcut_next.activated.connect(self._navigate_next)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate_previous(self) -> None:
        """Show the previous image in the results list."""
        if self._current_index > 0:
            self._current_index -= 1
            self._show_current_image()

    def _navigate_next(self) -> None:
        """Show the next image in the results list."""
        if self._current_index < len(self._results) - 1:
            self._current_index += 1
            self._show_current_image()

    # ------------------------------------------------------------------
    # File size formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in bytes to a human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    # ------------------------------------------------------------------
    # Face display info
    # ------------------------------------------------------------------

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
        """Return (rect, color, tooltip) for every non-deleted face."""
        if self._image_data is None:
            return []

        # Pre-resolve unidentified faces in one batch
        unidentified_ids = [
            f.id
            for f in self._image_data.faces
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
        for face in self._image_data.faces:
            if face.deleted_at is not None:
                continue

            rect = QtCore.QRectF(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h)
            color, tooltip = self._get_face_display_info(face)
            result.append((rect, color, tooltip))

        return result

    # ------------------------------------------------------------------
    # Image loading and display
    # ------------------------------------------------------------------

    def _load_image(self) -> None:
        """Load the current image file, draw face boxes, and display it."""
        if self._image_data is None:
            return

        file_path = Path(self._image_data.file_path)
        if not file_path.exists():
            self.image_label.setText(
                self.tr("Image file not found: {path}").format(path=file_path)
            )
            self._original_pixmap = None
            return

        reader = QtGui.QImageReader(str(file_path))
        reader.setAutoTransform(True)
        qimage = reader.read()

        if qimage.isNull():
            self.image_label.setText(
                self.tr("Failed to load image: {error}").format(
                    error=reader.errorString()
                )
            )
            self._original_pixmap = None
            return

        pixmap = QtGui.QPixmap.fromImage(qimage)

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

        tooltip_regions = [(rect, tooltip) for rect, _, tooltip in face_display]

        self.image_label.set_face_regions(
            pixmap_regions=tooltip_regions, pixmap_size=pixmap.size()
        )

        self._original_pixmap = pixmap
        self._update_image_display()

    def _update_image_display(self) -> None:
        """Scale the original pixmap to fit the scroll area viewport."""
        if self._original_pixmap is None:
            return

        available_size = self.scroll_area.viewport().size()
        scaled_pixmap = self._original_pixmap.scaled(
            available_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_browse_photo_folder_clicked(self) -> None:
        if self._image_data is not None:
            self.accept()
            self.navigate_to_browse.emit(str(self._image_data.file_path))

    def _on_label_faces_clicked(self) -> None:
        if self._image_data is not None:
            self.accept()
            self.navigate_to_labelling.emit(self._image_data.id)

    def _on_show_in_file_manager_clicked(self) -> None:
        if self._image_data is not None:
            reveal_in_file_manager(self._image_data.file_path)

    # ------------------------------------------------------------------
    # Geometry persistence and resize
    # ------------------------------------------------------------------

    def done(self, result: int) -> None:
        """Save geometry before closing."""
        self._map_widget.cleanup()
        save_widget_geometry(self, self._paths.window_state_file)
        super().done(result)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._resize_timer.start()
