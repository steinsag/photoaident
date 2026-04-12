import html
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import contains_eager, selectinload

from photoaident.core.search_person import find_best_person_for_face
from photoaident.db.cluster_means import deserialize_embedding, recompute_cluster_mean
from photoaident.ui.window_state import restore_widget_geometry, save_widget_geometry
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    Person,
)
from photoaident.db.vector_store import VectorStore
from photoaident.ui.widgets.cluster_table_widget import ClusterTableWidget
from photoaident.ui.widgets.face_crop_widget import FaceCropWidget
from photoaident.ui.widgets.image_preview_widget import ImagePreviewWidget
from photoaident.ui.widgets.new_person_dialog import NewPersonDialog
from photoaident.ui.widgets.person_list_widget import PersonListWidget

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.paths import AppPaths


@dataclass(frozen=True, slots=True)
class _FaceEntry:
    """Data for a single unidentified face shown in the labelling dialog."""

    face_id: int
    crop_path: Path
    image_path: Path
    bbox: tuple[int, int, int, int]
    taken_at: datetime | None


class LabellingDialog(QtWidgets.QDialog):
    """Modal dialog for labelling all unidentified faces of a single image."""

    def __init__(
        self,
        image_id: int,
        session_factory: "sessionmaker",
        paths: "AppPaths",
        vector_store: VectorStore,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Label Faces"))
        self.setMinimumSize(900, 700)

        self._image_id = image_id
        self._session_factory = session_factory
        self._paths = paths
        self._vector_store = vector_store

        self._face_data: list[_FaceEntry] = []
        self._current_idx: int | None = None
        self._query_embedding: np.ndarray | None = None
        self._selected_person: Person | None = None
        self._selected_cluster: EmbeddingCluster | None = None

        self._setup_ui()
        self._load_persons()
        self._load_faces()
        restore_widget_geometry(self, self._paths.window_state_file)

    @property
    def _current_face_id(self) -> int | None:
        """Return the face ID of the currently displayed face."""
        if self._current_idx is None or not self._face_data:
            return None
        return self._face_data[self._current_idx].face_id

    def _setup_ui(self) -> None:
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # --- Top row: image preview | face crop | action buttons ---
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setSpacing(8)

        self._image_preview = ImagePreviewWidget()
        self._image_preview.setMinimumSize(300, 300)
        top_layout.addWidget(self._image_preview, stretch=1)

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(6)

        self._face_info_label = QtWidgets.QLabel()
        self._face_info_label.setWordWrap(True)
        right_col.addWidget(self._face_info_label)
        right_col.addStretch()

        face_and_buttons_row = QtWidgets.QHBoxLayout()
        face_and_buttons_row.setSpacing(8)

        self._face_crop = FaceCropWidget()
        face_and_buttons_row.addWidget(self._face_crop)

        btn_col = QtWidgets.QVBoxLayout()
        btn_col.setSpacing(6)

        self.skip_btn = QtWidgets.QPushButton(self.tr("Skip Face"))
        self.skip_btn.clicked.connect(self._skip_face)
        btn_col.addWidget(self.skip_btn)

        self.anonymous_btn = QtWidgets.QPushButton(self.tr("Mark Anonymous"))
        self.anonymous_btn.clicked.connect(self._mark_anonymous)
        btn_col.addWidget(self.anonymous_btn)

        btn_col.addStretch()
        face_and_buttons_row.addLayout(btn_col)
        right_col.addLayout(face_and_buttons_row)
        top_layout.addLayout(right_col)

        root_layout.addLayout(top_layout)

        # --- Bottom: inline person selector ---
        person_group = QtWidgets.QGroupBox(self.tr("Select Person"))
        group_layout = QtWidgets.QHBoxLayout(person_group)
        group_layout.setSpacing(8)

        self._person_widget = PersonListWidget()
        self._person_widget.person_selected.connect(self._on_person_selected)
        self._person_widget.new_person_requested.connect(self._on_new_person_requested)
        group_layout.addWidget(self._person_widget)

        # Right panel: cluster table + confirm/cancel
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._cluster_widget = ClusterTableWidget()
        self._cluster_widget.cluster_selected.connect(self._on_cluster_selected)
        right_layout.addWidget(self._cluster_widget, stretch=1)

        confirm_row = QtWidgets.QHBoxLayout()
        self._cancel_btn = QtWidgets.QPushButton(self.tr("Cancel"))
        self._cancel_btn.clicked.connect(self._on_cancel)
        confirm_row.addWidget(self._cancel_btn)

        self.confirm_btn = QtWidgets.QPushButton(self.tr("Confirm"))
        self.confirm_btn.clicked.connect(self._on_confirm)
        confirm_row.addWidget(self.confirm_btn)
        confirm_row.addStretch()

        right_layout.addLayout(confirm_row)
        group_layout.addWidget(right_panel, stretch=1)

        root_layout.addWidget(person_group)

        # Close button row
        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch()
        close_btn = QtWidgets.QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root_layout.addLayout(close_row)

        self._set_buttons_enabled(False)
        self._update_confirm_button()

    # ------------------------------------------------------------------
    # Backward-compatible aliases for internal widget access (used by tests)
    # ------------------------------------------------------------------

    @property
    def _person_list(self) -> "QtWidgets.QListWidget":
        """Return the inner QListWidget of the person panel."""
        return self._person_widget._list_widget

    @property
    def _search_edit(self) -> "QtWidgets.QLineEdit":
        """Return the search field of the person panel."""
        return self._person_widget._search_edit

    @property
    def _cluster_table(self) -> "QtWidgets.QTableWidget":
        """Return the inner QTableWidget of the cluster panel."""
        return self._cluster_widget._table

    @property
    def _crop_label(self) -> FaceCropWidget:
        """Return the face crop widget."""
        return self._face_crop

    # ------------------------------------------------------------------
    # Private: face loading pipeline
    # ------------------------------------------------------------------

    def _load_faces(self) -> None:
        """Load all unidentified faces for the image and show the first one."""
        with self._session_factory() as session:
            stmt = (
                select(Face)
                .join(Face.image)
                .outerjoin(Image.metadata_rel)
                .options(contains_eager(Face.image).contains_eager(Image.metadata_rel))
                .where(
                    Face.image_id == self._image_id,
                    Face.state == FaceState.UNIDENTIFIED,
                    Face.deleted_at.is_(None),
                )
                .order_by(Face.bbox_x, Face.bbox_y)
            )
            faces = session.execute(stmt).unique().scalars().all()
            self._face_data = [
                _FaceEntry(
                    face_id=f.id,
                    crop_path=self._paths.face_crops_dir / f"{f.id}.jpg",
                    image_path=Path(f.image.file_path),
                    bbox=(f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h),
                    taken_at=(
                        f.image.metadata_rel.taken_at if f.image.metadata_rel else None
                    ),
                )
                for f in faces
            ]

        if not self._face_data:
            self._show_completion_state()
            return

        self._current_idx = 0
        self._show_face(0)

    def _show_face(self, idx: int) -> None:
        """Display the face at idx in the preview/crop widgets."""
        self._current_idx = idx
        entry = self._face_data[idx]

        # Fetch embedding for similarity scoring
        self._query_embedding = None
        try:
            self._query_embedding = self._vector_store.get_embedding(entry.face_id)
        except Exception:
            logger.warning(
                "Failed to retrieve embedding for face %s",
                entry.face_id,
                exc_info=True,
            )

        total = len(self._face_data)
        face_num = idx + 1

        if entry.taken_at is not None:
            taken_str = entry.taken_at.strftime("%Y-%m-%d %H:%M")
        else:
            taken_str = self.tr("Unknown date")
        path_escaped = html.escape(str(entry.image_path))
        taken_escaped = html.escape(taken_str)
        progress_escaped = html.escape(
            self.tr("Face {current} of {total}").format(current=face_num, total=total)
        )
        label_file_path = html.escape(self.tr("File Path"))
        label_taken_at = html.escape(self.tr("Taken At"))
        self._face_info_label.setText(
            f"<b>{label_file_path}:</b> {path_escaped}<br>"
            f"<b>{label_taken_at}:</b> {taken_escaped}<br>"
            f"<b>{progress_escaped}</b>"
        )

        self._image_preview.load(entry.image_path, entry.bbox)
        self._face_crop.load(entry.crop_path, entry.image_path, entry.bbox)
        self._set_buttons_enabled(True)
        self._person_widget.clear_selection()
        self._cluster_widget.clear_data()
        self._search_edit.clear()
        best_person_id = self._find_best_person_id()
        if best_person_id is not None:
            self._person_widget.select_by_id(best_person_id)
        # Unconditionally sync _selected_person from the widget and refresh
        # cluster scores for this face.  Qt suppresses currentItemChanged when
        # the same item is re-selected, so we cannot rely on the signal alone
        # when the best-match person is the same across consecutive faces.
        self._selected_person = self._person_widget.current_person()
        self._refresh_cluster_selection()

    def _advance_to_next(self) -> None:
        """Remove the current face from the list and show the next unidentified face."""
        if self._current_idx is None or not self._face_data:
            return
        del self._face_data[self._current_idx]
        remaining = len(self._face_data)
        if remaining == 0:
            self._show_completion_state()
        else:
            self._show_face(min(self._current_idx, remaining - 1))

    # ------------------------------------------------------------------
    # Private: person list
    # ------------------------------------------------------------------

    def _load_persons(self) -> None:
        """Load all persons from DB, preserving filter text and current selection."""
        prev_person_id = (
            self._selected_person.id if self._selected_person is not None else None
        )
        self._selected_person = None
        self._selected_cluster = None
        with self._session_factory() as session:
            persons = (
                session.execute(
                    select(Person)
                    .options(selectinload(Person.clusters))
                    .order_by(Person.name)
                )
                .scalars()
                .all()
            )
            session.expunge_all()
            self._person_widget.set_persons(list(persons))
        if prev_person_id is not None:
            self._person_widget.select_by_id(prev_person_id)
        self._update_confirm_button()

    def _on_person_selected(
        self,
        person: Person | None,
        _: object = None,
    ) -> None:
        """Handle person selection from PersonListWidget signal or direct call."""
        self._selected_person = person
        self._populate_clusters_for_person(person)

    def _populate_clusters_for_person(self, person: Person | None) -> None:
        """Populate the cluster widget for *person*, auto-selecting the best match."""
        self._selected_cluster = None
        if person is None:
            self._cluster_widget.clear_data()
        else:
            cluster_by_age: dict[str, EmbeddingCluster] = {
                c.age_group: c for c in person.clusters if c.age_group is not None
            }
            scores = (
                self._compute_cluster_scores(cluster_by_age)
                if self._query_embedding is not None
                else {}
            )
            best_row = self._cluster_widget.populate(cluster_by_age, scores)
            if best_row is not None:
                self._cluster_widget.select_row(best_row)
        self._update_confirm_button()

    def _refresh_cluster_selection(self) -> None:
        """Recalculate cluster scores for the current face; no DB round-trip."""
        self._populate_clusters_for_person(self._selected_person)

    def _on_cluster_selected(self, cluster: EmbeddingCluster | None) -> None:
        self._selected_cluster = cluster
        self._update_confirm_button()

    def _on_new_person_requested(self) -> None:
        dlg = NewPersonDialog(self._session_factory, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        new_person_id = dlg.created_person_id()
        if new_person_id is None:
            return

        with self._session_factory() as session:
            loaded = session.execute(
                select(Person)
                .where(Person.id == new_person_id)
                .options(selectinload(Person.clusters))
            ).scalar_one()
            session.expunge_all()

        self._person_widget.add_person_sorted(loaded)
        self._person_widget.select_by_id(new_person_id)

    def _find_best_person_id(self) -> int | None:
        """Return the person_id whose cluster mean best matches the current face."""
        if self._current_face_id is None:
            return None
        return find_best_person_for_face(
            self._current_face_id, self._session_factory, self._vector_store
        )

    def _compute_cluster_scores(
        self, cluster_by_age: dict[str, EmbeddingCluster]
    ) -> dict[str, float]:
        """Compute cosine similarity between query_embedding and each cluster mean.

        Reads persisted mean embeddings from the DB — no FAISS access needed.
        """
        assert self._query_embedding is not None

        q = self._query_embedding.astype(VectorStore.EMBEDDING_DTYPE).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return {}
        q = q / q_norm

        scores: dict[str, float] = {}
        for age_key, cluster in cluster_by_age.items():
            if cluster.mean_embedding is None:
                continue
            mean_vec = deserialize_embedding(cluster.mean_embedding)
            if mean_vec is None:
                continue
            scores[age_key] = float(np.dot(q, mean_vec))

        return scores

    def _update_confirm_button(self) -> None:
        self.confirm_btn.setEnabled(
            self._selected_person is not None and self._selected_cluster is not None
        )

    # ------------------------------------------------------------------
    # Private: UI helpers
    # ------------------------------------------------------------------

    def _show_completion_state(self) -> None:
        """Disable action widgets and show a completion message."""
        self._current_idx = None
        self._query_embedding = None
        self._face_info_label.clear()
        self._image_preview.clear()
        self._face_crop.clear()
        self._face_crop.setText(self.tr("All done! No unidentified faces remain."))
        self._person_widget.set_persons([])
        self._selected_person = None
        self._selected_cluster = None
        self._cluster_widget.clear_data()
        self._set_buttons_enabled(False)
        self._update_confirm_button()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.skip_btn.setEnabled(enabled)
        self.anonymous_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Private: actions
    # ------------------------------------------------------------------

    def _on_confirm(self) -> None:
        """Assign the current face to the selected person/cluster."""
        if self._current_face_id is None:
            return
        if self._selected_person is None or self._selected_cluster is None:
            return
        person = self._selected_person
        cluster = self._selected_cluster
        face_id = self._current_face_id
        with self._session_factory() as session:
            face = session.get(Face, face_id)
            if face is not None:
                face.state = FaceState.IDENTIFIED
                face.person_id = person.id
                face.cluster_id = cluster.id
                face.labelled_at = datetime.now(timezone.utc)
                session.commit()
        recompute_cluster_mean(cluster.id, self._session_factory, self._vector_store)
        self._advance_to_next()

    def _on_cancel(self) -> None:
        """Clear the current person/cluster selection."""
        self._selected_person = None
        self._selected_cluster = None
        self._person_widget.clear_selection()
        self._cluster_widget.clear_data()
        self._update_confirm_button()

    def _mark_anonymous(self) -> None:
        if self._current_face_id is None:
            return
        face_id = self._current_face_id
        with self._session_factory() as session:
            face = session.get(Face, face_id)
            if face is not None:
                face.state = FaceState.ANONYMOUS
                face.labelled_at = datetime.now(timezone.utc)
                session.commit()
        self._advance_to_next()

    def _skip_face(self) -> None:
        if self._current_idx is None:
            return
        next_idx = self._current_idx + 1
        if next_idx < len(self._face_data):
            self._show_face(next_idx)
        else:
            self._show_completion_state()

    def done(self, result: int) -> None:
        """Save geometry before closing."""
        save_widget_geometry(self, self._paths.window_state_file)
        super().done(result)

    # ------------------------------------------------------------------
    # Backward-compatible method aliases
    # ------------------------------------------------------------------

    def _filter_persons(self, text: str) -> None:
        """Apply person filter on the person list widget."""
        self._person_widget._apply_filter(text)

    def _create_new_person(self) -> None:
        """Alias for _on_new_person_requested (used in tests)."""
        self._on_new_person_requested()
