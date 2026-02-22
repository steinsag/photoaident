from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from PySide6 import QtCore, QtWidgets
from sqlalchemy import func, select
from sqlalchemy.orm import contains_eager

from photoaident.db.database import Face, FaceState, Image, ImageMetadata
from photoaident.ui.widgets.assign_person_dialog import AssignPersonDialog
from photoaident.ui.widgets.face_crop import FaceCropWidget

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.paths import AppPaths


class LabellingPage(QtWidgets.QWidget):
    """Page for labelling unidentified faces one by one."""

    def __init__(
        self,
        session_factory: "sessionmaker",
        paths: "AppPaths",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.session_factory = session_factory
        self.paths = paths
        self._current_face_id: Optional[int] = None
        self._skipped: set[int] = set()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.face_crop = FaceCropWidget()
        layout.addWidget(self.face_crop, stretch=1)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.addStretch()

        self.assign_btn = QtWidgets.QPushButton(self.tr("Assign to Person\u2026"))
        self.assign_btn.clicked.connect(self._assign_face)
        action_layout.addWidget(self.assign_btn)

        self.anonymous_btn = QtWidgets.QPushButton(self.tr("Mark Anonymous"))
        self.anonymous_btn.clicked.connect(self._mark_anonymous)
        action_layout.addWidget(self.anonymous_btn)

        self.skip_btn = QtWidgets.QPushButton(self.tr("Skip"))
        self.skip_btn.clicked.connect(self._skip_face)
        action_layout.addWidget(self.skip_btn)

        action_layout.addStretch()
        layout.addLayout(action_layout)

        self._set_buttons_enabled(False)

    def refresh(self) -> None:
        """Reload count + first face. Called when page becomes visible."""
        self._skipped.clear()
        self._load_next_face()

    def _load_next_face(self) -> None:
        # Count total unidentified faces for status / empty-state messages
        with self.session_factory() as session:
            total_unidentified: int = (
                session.scalar(
                    select(func.count(Face.id)).where(
                        Face.state == FaceState.UNIDENTIFIED,
                        Face.deleted_at.is_(None),
                    )
                )
                or 0
            )

        # Build the query for the next non-skipped unidentified face
        stmt = (
            select(Face)
            .join(Face.image)
            .outerjoin(Image.metadata_rel)
            .options(contains_eager(Face.image).contains_eager(Image.metadata_rel))
            .where(
                Face.state == FaceState.UNIDENTIFIED,
                Face.deleted_at.is_(None),
            )
            .order_by(
                ImageMetadata.taken_at.asc().nulls_last(),
                Image.indexed_at.asc(),
            )
            .limit(1)
        )
        if self._skipped:
            stmt = stmt.where(Face.id.not_in(list(self._skipped)))

        face_data: Optional[tuple[int, Path, Optional[Path], str, float]] = None
        with self.session_factory() as session:
            face = session.execute(stmt).unique().scalar_one_or_none()
            if face is not None:
                face_id = face.id
                crop_path = self.paths.face_crops_dir / f"{face_id}.jpg"
                thumb_path = (
                    self.paths.thumbs_dir / f"{face.image.file_hash}.jpg"
                    if face.image.file_hash
                    else None
                )
                meta = face.image.metadata_rel
                if meta is not None and meta.taken_at is not None:
                    taken_at = meta.taken_at.strftime("%Y-%m-%d")
                else:
                    taken_at = self.tr("Unknown")
                confidence = face.detection_confidence
                face_data = (face_id, crop_path, thumb_path, taken_at, confidence)

        if face_data is None:
            self._current_face_id = None
            self._show_empty_state(total_unidentified)
            return

        face_id, crop_path, thumb_path, taken_at, confidence = face_data
        self._current_face_id = face_id
        self.status_label.setText(
            self.tr("{count} face(s) remaining").format(count=total_unidentified)
        )
        self.face_crop.load(
            crop_path=crop_path,
            thumb_path=thumb_path,
            taken_at=taken_at,
            confidence=confidence,
        )
        self._set_buttons_enabled(True)

    def _show_empty_state(self, total_unidentified: int) -> None:
        if total_unidentified == 0:
            msg = self.tr("All done! No unidentified faces remain.")
        else:
            msg = self.tr(
                "All remaining faces skipped this session. "
                "Restart the app to review them again."
            )
        self.status_label.setText(msg)
        self.face_crop.clear()
        self._set_buttons_enabled(False)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.assign_btn.setEnabled(enabled)
        self.anonymous_btn.setEnabled(enabled)
        self.skip_btn.setEnabled(enabled)

    def _assign_face(self) -> None:
        if self._current_face_id is None:
            return
        dialog = AssignPersonDialog(self.session_factory, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        result = dialog.result_person_cluster()
        if result is None:
            return
        person, cluster = result
        with self.session_factory() as session:
            face = session.get(Face, self._current_face_id)
            if face is not None:
                face.state = FaceState.IDENTIFIED
                face.person_id = person.id
                face.cluster_id = cluster.id
                face.labelled_at = datetime.now(timezone.utc)
                session.commit()
        self._load_next_face()

    def _mark_anonymous(self) -> None:
        if self._current_face_id is None:
            return
        with self.session_factory() as session:
            face = session.get(Face, self._current_face_id)
            if face is not None:
                face.state = FaceState.ANONYMOUS
                face.labelled_at = datetime.now(timezone.utc)
                session.commit()
        self._load_next_face()

    def _skip_face(self) -> None:
        if self._current_face_id is None:
            return
        self._skipped.add(self._current_face_id)
        self._load_next_face()
