from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from photoaident.db.cluster_means import recompute_cluster_mean
from photoaident.db.database import (
    AGE_CLUSTERS,
    EmbeddingCluster,
    Face,
    FaceState,
    Person,
)
from photoaident.ui.widgets.new_person_dialog import NewPersonDialog

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from photoaident.db.vector_store import VectorStore
    from photoaident.paths import AppPaths


class _PendingKind(Enum):
    REMOVE = auto()
    MOVE = auto()


@dataclass
class _PendingChange:
    kind: _PendingKind
    new_cluster_id: int | None  # only for MOVE


class ReferenceFaceWidget(QtWidgets.QWidget):
    """Small tile showing one identified face with remove/move actions."""

    remove_requested = QtCore.Signal(int)  # face_id
    move_requested = QtCore.Signal(int, int)  # face_id, target_cluster_id

    def __init__(
        self,
        face_id: int,
        crop_path: Path,
        cluster_id: int,
        other_clusters: list[tuple[int, str]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._face_id = face_id
        self._cluster_id = cluster_id
        self._other_clusters = other_clusters
        self.setFixedWidth(140)
        self._setup_ui(crop_path)

    def _setup_ui(self, crop_path: Path) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Face crop image
        self._image_label = QtWidgets.QLabel()
        self._image_label.setFixedSize(120, 120)
        self._image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pixmap = QtGui.QPixmap(str(crop_path))
        if pixmap.isNull():
            self._image_label.setText("?")
        else:
            self._image_label.setPixmap(
                pixmap.scaled(
                    120,
                    120,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )
        layout.addWidget(self._image_label)

        # Pending action text (hidden by default)
        self._status_label = QtWidgets.QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # Remove / Undo button
        self._remove_btn = QtWidgets.QPushButton(self.tr("Remove"))
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        layout.addWidget(self._remove_btn)

        # Move to… button
        self._move_btn = QtWidgets.QPushButton(self.tr("Move to\u2026"))
        self._move_btn.setEnabled(bool(self._other_clusters))
        self._move_btn.clicked.connect(self._on_move_clicked)
        layout.addWidget(self._move_btn)

    def set_pending(self, change: _PendingChange | None) -> None:
        """Update the visual state to reflect a staged change (or clear it)."""
        if change is None:
            self._status_label.setText("")
            self._status_label.setVisible(False)
            self._remove_btn.setText(self.tr("Remove"))
            self.setStyleSheet("")
        elif change.kind == _PendingKind.REMOVE:
            self._status_label.setText(self.tr("Will be removed"))
            self._status_label.setVisible(True)
            self._remove_btn.setText(self.tr("Undo"))
            self.setStyleSheet("background-color: #ffeeba;")
        elif change.kind == _PendingKind.MOVE:
            # Find the label of the target cluster
            target_label = ""
            for cid, clabel in self._other_clusters:
                if cid == change.new_cluster_id:
                    target_label = clabel
                    break
            self._status_label.setText(
                self.tr("Will move to: {cluster}").format(cluster=target_label)
            )
            self._status_label.setVisible(True)
            self._remove_btn.setText(self.tr("Remove"))
            self.setStyleSheet("background-color: #c3e6cb;")

    def _on_remove_clicked(self) -> None:
        self.remove_requested.emit(self._face_id)

    def _on_move_clicked(self) -> None:
        menu = QtWidgets.QMenu(self)
        for cluster_id, label in self._other_clusters:
            action = menu.addAction(label)
            action.setData(cluster_id)
        chosen = menu.exec(
            self._move_btn.mapToGlobal(self._move_btn.rect().bottomLeft())
        )
        if chosen is not None:
            self.move_requested.emit(self._face_id, chosen.data())


class PersonsPage(QtWidgets.QWidget):
    """Page for reviewing and correcting reference faces per person."""

    def __init__(
        self,
        session_factory: "sessionmaker",
        paths: "AppPaths",
        vector_store: "VectorStore",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session_factory = session_factory
        self.paths = paths
        self.vector_store = vector_store
        self._selected_person_id: int | None = None
        self._pending: dict[int, _PendingChange] = {}
        self._pending_name: str | None = None
        self._current_person_name: str | None = None
        self._face_widgets: dict[int, ReferenceFaceWidget] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)

        # ── Left panel ──────────────────────────────────────────────────────
        left = QtWidgets.QWidget()
        left.setFixedWidth(220)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self._filter_edit = QtWidgets.QLineEdit()
        self._filter_edit.setPlaceholderText(self.tr("Search persons\u2026"))
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        left_layout.addWidget(self._filter_edit)

        self._person_list = QtWidgets.QListWidget()
        self._person_list.currentItemChanged.connect(self._on_person_selected)
        left_layout.addWidget(self._person_list)

        self._new_person_btn = QtWidgets.QPushButton(self.tr("New Person\u2026"))
        self._new_person_btn.clicked.connect(self._on_new_person)
        left_layout.addWidget(self._new_person_btn)

        self._delete_person_btn = QtWidgets.QPushButton(self.tr("Delete Person\u2026"))
        self._delete_person_btn.clicked.connect(self._on_delete_person)
        self._delete_person_btn.setEnabled(False)
        left_layout.addWidget(self._delete_person_btn)

        splitter.addWidget(left)

        # ── Right panel ─────────────────────────────────────────────────────
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self._person_name_edit = QtWidgets.QLineEdit()
        font = self._person_name_edit.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        self._person_name_edit.setFont(font)
        self._person_name_edit.setVisible(False)
        self._person_name_edit.textEdited.connect(self._on_name_edited)
        right_layout.addWidget(self._person_name_edit)

        # Placeholder shown when no person is selected
        self._placeholder_label = QtWidgets.QLabel(
            self.tr("Select a person to view their reference faces.")
        )
        self._placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self._placeholder_label)

        # Scroll area for cluster groups
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setVisible(False)
        self._clusters_widget = QtWidgets.QWidget()
        self._clusters_layout = QtWidgets.QVBoxLayout(self._clusters_widget)
        self._clusters_layout.addStretch()
        self._scroll.setWidget(self._clusters_widget)
        right_layout.addWidget(self._scroll, stretch=1)

        # Bottom action bar
        bottom_bar = QtWidgets.QHBoxLayout()
        self._changes_label = QtWidgets.QLabel()
        bottom_bar.addWidget(self._changes_label)
        bottom_bar.addStretch()

        self._cancel_btn = QtWidgets.QPushButton(self.tr("Cancel"))
        self._cancel_btn.clicked.connect(self._cancel)
        self._cancel_btn.setEnabled(False)
        bottom_bar.addWidget(self._cancel_btn)

        self._confirm_btn = QtWidgets.QPushButton(self.tr("Confirm"))
        self._confirm_btn.clicked.connect(self._confirm)
        self._confirm_btn.setEnabled(False)
        bottom_bar.addWidget(self._confirm_btn)

        right_layout.addLayout(bottom_bar)
        splitter.addWidget(right)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

    def refresh(self) -> None:
        """Reload persons list; re-select the same person if still present."""
        selected_id = self._selected_person_id
        self._load_persons()
        # Try to re-select the previously selected person
        if selected_id is not None:
            for i in range(self._person_list.count()):
                item = self._person_list.item(i)
                if (
                    item is not None
                    and item.data(QtCore.Qt.ItemDataRole.UserRole) == selected_id
                ):
                    self._person_list.setCurrentItem(item)
                    return
        # Nothing found — clear right panel
        self._selected_person_id = None
        self._clear_right_panel()

    def _load_persons(self) -> None:
        with self.session_factory() as session:
            persons = (
                session.execute(select(Person).order_by(Person.name)).scalars().all()
            )
            person_data = [(p.id, p.name) for p in persons]

        self._person_list.blockSignals(True)
        self._person_list.clear()
        for person_id, name in person_data:
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, person_id)
            self._person_list.addItem(item)
        self._person_list.blockSignals(False)

        # Re-apply current filter
        self._on_filter_changed(self._filter_edit.text())

    def _on_new_person(self) -> None:
        dlg = NewPersonDialog(self.session_factory, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        new_person_id = dlg.created_person_id()
        if new_person_id is None:
            # Dialog was accepted but no person was actually created;
            # preserve current UI state.
            return
        self._filter_edit.clear()  # ensure new person isn't hidden by an active filter
        self._load_persons()
        for i in range(self._person_list.count()):
            item = self._person_list.item(i)
            if (
                item is not None
                and item.data(QtCore.Qt.ItemDataRole.UserRole) == new_person_id
            ):
                self._person_list.setCurrentItem(item)
                break

    def _on_delete_person(self) -> None:
        if self._selected_person_id is None or self._current_person_name is None:
            return
        display_name = self._pending_name or self._current_person_name
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(self.tr("Delete Person"))
        msg.setText(self.tr('Delete "{name}"?').format(name=display_name))
        msg.setInformativeText(
            self.tr(
                "This will permanently remove the person. "
                "All their labelled faces will be marked as unknown. "
                "This action cannot be undone."
            )
        )
        msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        delete_btn = msg.addButton(
            self.tr("Delete"), QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        msg.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg.exec()
        if msg.clickedButton() is not delete_btn:
            return
        self._delete_person(self._selected_person_id)

    def _delete_person(self, person_id: int) -> None:
        """Unlink all faces and delete the person with their clusters from the DB."""
        with self.session_factory() as session:
            person = session.get(Person, person_id)
            if person is None:
                return
            # Unlink all identified faces so they return to the labelling queue
            for face in person.faces:
                face.state = FaceState.UNIDENTIFIED
                face.person_id = None
                face.cluster_id = None
                face.labelled_at = None
            session.delete(person)
            session.commit()

        self._selected_person_id = None
        self._load_persons()
        self._clear_right_panel()

    def _on_filter_changed(self, text: str) -> None:
        needle = text.lower().strip()
        for i in range(self._person_list.count()):
            item = self._person_list.item(i)
            if item is None:
                continue
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _on_person_selected(
        self,
        current: QtWidgets.QListWidgetItem | None,
        _previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        if current is None:
            self._selected_person_id = None
            self._clear_right_panel()
            return
        person_id: int = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._load_person(person_id)

    def _clear_right_panel(self) -> None:
        self._person_name_edit.setVisible(False)
        self._scroll.setVisible(False)
        self._placeholder_label.setVisible(True)
        self._pending.clear()
        self._pending_name = None
        self._current_person_name = None
        self._face_widgets.clear()
        self._delete_person_btn.setEnabled(False)
        self._update_action_buttons()

    def _query_person_and_clusters(
        self,
        session: "Session",
        person_id: int,
    ) -> tuple[str, list[tuple[int, str, list[tuple[int, Path]]]]] | None:
        """Query person name and per-cluster face data from DB.

        Returns ``(person_name, cluster_data)`` where each entry in
        ``cluster_data`` is ``(cluster_id, label, [(face_id, crop_path), ...])``,
        or ``None`` if the person no longer exists.
        """
        person = session.get(Person, person_id)
        if person is None:
            return None
        person_name = person.name

        clusters = (
            session.execute(
                select(EmbeddingCluster)
                .where(EmbeddingCluster.person_id == person_id)
                .options(selectinload(EmbeddingCluster.faces))
            )
            .scalars()
            .all()
        )

        display_labels = self._age_display_labels()
        cluster_id_to_label: dict[int, str] = {}
        for cluster in clusters:
            age_key = cluster.age_group or cluster.label or ""
            label = display_labels.get(age_key, age_key)
            cluster_id_to_label[cluster.id] = label

        cluster_data: list[tuple[int, str, list[tuple[int, Path]]]] = []
        for age_key in AGE_CLUSTERS:
            matched = [c for c in clusters if c.age_group == age_key]
            if not matched:
                continue
            cluster = matched[0]
            label = cluster_id_to_label[cluster.id]
            ref_faces = [
                f
                for f in cluster.faces
                if f.state == FaceState.IDENTIFIED and f.deleted_at is None
            ]
            face_paths = [
                (f.id, self.paths.face_crops_dir / f"{f.id}.jpg") for f in ref_faces
            ]
            cluster_data.append((cluster.id, label, face_paths))

        return person_name, cluster_data

    def _load_person(self, person_id: int) -> None:
        self._selected_person_id = person_id
        self._pending.clear()
        self._face_widgets.clear()

        with self.session_factory() as session:
            result = self._query_person_and_clusters(session, person_id)
            if result is None:
                self._clear_right_panel()
                return
            person_name, cluster_data = result

        self._current_person_name = person_name
        self._pending_name = None
        self._person_name_edit.setText(person_name)
        self._person_name_edit.setVisible(True)
        self._delete_person_btn.setEnabled(True)
        self._placeholder_label.setVisible(False)
        self._scroll.setVisible(True)

        # Rebuild clusters widget
        # Remove all existing widgets from layout except the trailing stretch
        while self._clusters_layout.count() > 1:
            layout_item = self._clusters_layout.takeAt(0)
            if layout_item is not None:
                w = layout_item.widget()
                if w is not None:
                    w.deleteLater()

        for cluster_id, label, face_paths in cluster_data:
            # Other clusters for the move menu = all clusters except current
            other_clusters = [
                (cid, clabel)
                for cid, clabel in [(d[0], d[1]) for d in cluster_data]
                if cid != cluster_id
            ]
            group = self._build_cluster_section(
                cluster_id, label, face_paths, other_clusters
            )
            self._clusters_layout.insertWidget(self._clusters_layout.count() - 1, group)

        self._update_action_buttons()

    def _build_cluster_section(
        self,
        cluster_id: int,
        label: str,
        face_paths: list[tuple[int, Path]],
        other_clusters: list[tuple[int, str]],
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(label)
        h_layout = QtWidgets.QHBoxLayout(group)
        h_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        if not face_paths:
            placeholder = QtWidgets.QLabel(self.tr("(No faces)"))
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            h_layout.addWidget(placeholder)
        else:
            for face_id, crop_path in face_paths:
                widget = ReferenceFaceWidget(
                    face_id=face_id,
                    crop_path=crop_path,
                    cluster_id=cluster_id,
                    other_clusters=other_clusters,
                )
                widget.remove_requested.connect(self._on_remove_requested)
                widget.move_requested.connect(self._on_move_requested)
                self._face_widgets[face_id] = widget
                h_layout.addWidget(widget)

        return group

    def _on_remove_requested(self, face_id: int) -> None:
        if (
            face_id in self._pending
            and self._pending[face_id].kind == _PendingKind.REMOVE
        ):
            # Undo
            del self._pending[face_id]
            change = None
        else:
            change = _PendingChange(_PendingKind.REMOVE, None)
            self._pending[face_id] = change

        widget = self._face_widgets.get(face_id)
        if widget is not None:
            widget.set_pending(change)

        self._update_action_buttons()

    def _on_move_requested(self, face_id: int, cluster_id: int) -> None:
        change = _PendingChange(_PendingKind.MOVE, cluster_id)
        self._pending[face_id] = change

        widget = self._face_widgets.get(face_id)
        if widget is not None:
            widget.set_pending(change)

        self._update_action_buttons()

    def _on_name_edited(self, text: str) -> None:
        stripped = text.strip()
        if stripped != text:
            self._person_name_edit.blockSignals(True)
            self._person_name_edit.setText(stripped)
            self._person_name_edit.blockSignals(False)
        # Track any deviation from the saved name, including empty string.
        # _pending_name="" means the field was cleared (dirty but invalid).
        if stripped != self._current_person_name:
            self._pending_name = stripped
        else:
            self._pending_name = None
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        n = len(self._pending)
        name_is_dirty = self._pending_name is not None
        # Empty _pending_name means the field was cleared — dirty but not committable.
        name_is_valid = self._pending_name is None or bool(self._pending_name)
        total = n + (1 if name_is_dirty else 0)
        has_pending = total > 0
        self._confirm_btn.setEnabled(has_pending and name_is_valid)
        self._cancel_btn.setEnabled(has_pending)
        if has_pending:
            self._changes_label.setText(
                self.tr("{n} pending change(s)").format(n=total)
            )
        else:
            self._changes_label.setText("")

    @staticmethod
    def _apply_face_change(face: Face, change: _PendingChange) -> None:
        """Apply a pending change to a face object within an open session."""
        if change.kind == _PendingKind.REMOVE:
            face.state = FaceState.UNIDENTIFIED
            face.person_id = None
            face.cluster_id = None
            face.labelled_at = None
        elif change.kind == _PendingKind.MOVE:
            face.cluster_id = change.new_cluster_id

    @staticmethod
    def _affected_cluster_ids_for_change(
        face: Face, change: _PendingChange
    ) -> set[int]:
        """Return the cluster IDs that need their mean recomputed after this change."""
        ids: set[int] = set()
        if face.cluster_id is not None:
            ids.add(face.cluster_id)
        if change.kind == _PendingKind.MOVE and change.new_cluster_id is not None:
            ids.add(change.new_cluster_id)
        return ids

    def _persist_pending_changes(self) -> tuple[bool, set[int]]:
        """Write all staged changes to the DB in a single transaction.

        Returns ``(name_changed, affected_cluster_ids)`` so the caller can
        refresh the persons list and recompute cluster means as needed.
        """
        name_changed = False
        affected_cluster_ids: set[int] = set()

        with self.session_factory() as session:
            if self._pending_name and self._selected_person_id is not None:
                person = session.get(Person, self._selected_person_id)
                if person is not None:
                    person.name = self._pending_name
                    name_changed = True

            for face_id, change in self._pending.items():
                face: Face | None = session.get(Face, face_id)
                if face is None:
                    continue
                affected_cluster_ids |= self._affected_cluster_ids_for_change(
                    face, change
                )
                self._apply_face_change(face, change)

            session.commit()

        return name_changed, affected_cluster_ids

    def _confirm(self) -> None:
        if not self._pending and self._pending_name is None:
            return

        name_changed, affected_cluster_ids = self._persist_pending_changes()

        for cluster_id in affected_cluster_ids:
            recompute_cluster_mean(cluster_id, self.session_factory, self.vector_store)

        self._pending.clear()
        self._pending_name = None

        if name_changed:
            self.refresh()
        elif self._selected_person_id is not None:
            self._load_person(self._selected_person_id)
        else:
            self._update_action_buttons()

    def _cancel(self) -> None:
        for widget in self._face_widgets.values():
            widget.set_pending(None)
        self._pending.clear()
        self._pending_name = None
        if self._current_person_name is not None:
            self._person_name_edit.setText(self._current_person_name)
        self._update_action_buttons()

    def _age_display_labels(self) -> dict[str, str]:
        return {
            "infant": self.tr("Infant (0\u20133)"),
            "youngster": self.tr("Youngster (4\u201312)"),
            "teenager": self.tr("Teenager (13\u201319)"),
            "adult": self.tr("Adult (20\u201375)"),
            "senior": self.tr("Senior (75+)"),
        }
