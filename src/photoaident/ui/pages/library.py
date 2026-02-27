from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select

from photoaident.core.search import find_images_by_person
from photoaident.db.database import Image, Person
from photoaident.ui.widgets.thumbnail_grid import ThumbnailGrid

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore
    from photoaident.paths import AppPaths


class LibraryPage(QtWidgets.QWidget):
    """Page showing all indexed images with filtering by person."""

    def __init__(
        self,
        session_factory: "sessionmaker",
        paths: "AppPaths",
        vector_store: "VectorStore | None" = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session_factory = session_factory
        self.paths = paths
        self.vector_store = vector_store

        layout = QtWidgets.QVBoxLayout(self)

        # Filter bar: toggle button
        filter_bar = QtWidgets.QHBoxLayout()
        self.person_filter_btn = QtWidgets.QPushButton(self.tr("Filter by Person"))
        self.person_filter_btn.setCheckable(True)
        self.person_filter_btn.setEnabled(False)
        self.person_filter_btn.toggled.connect(self._on_filter_btn_toggled)
        filter_bar.addWidget(self.person_filter_btn)
        filter_bar.addStretch()
        layout.addLayout(filter_bar)

        # Collapsible filter panel
        self.filter_panel = QtWidgets.QFrame()
        self.filter_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.filter_panel.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.filter_panel.setVisible(False)
        panel_layout = QtWidgets.QVBoxLayout(self.filter_panel)

        search_row = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText(self.tr("Search persons…"))
        self.search_edit.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self.search_edit)
        self.deselect_btn = QtWidgets.QPushButton(self.tr("Deselect All"))
        self.deselect_btn.clicked.connect(self._deselect_all)
        search_row.addWidget(self.deselect_btn)
        panel_layout.addLayout(search_row)

        self.person_list_widget = QtWidgets.QListWidget()
        self.person_list_widget.setMaximumHeight(200)
        self.person_list_widget.itemChanged.connect(self._on_person_filter_changed)
        panel_layout.addWidget(self.person_list_widget)

        layout.addWidget(self.filter_panel)

        # Image grid
        self.grid = ThumbnailGrid(self.session_factory)
        self.grid.navigate_to_labelling.connect(self._on_navigate_to_labelling)
        layout.addWidget(self.grid)

        self._populate_person_list()
        self.load_images()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._populate_person_list()

    def _on_filter_btn_toggled(self, checked: bool) -> None:
        self.filter_panel.setVisible(checked)
        if checked:
            self._populate_person_list()
            self.search_edit.setFocus()
            self.search_edit.selectAll()

    def _populate_person_list(self) -> None:
        checked = set(self._selected_person_ids())
        self.person_list_widget.blockSignals(True)
        self.person_list_widget.clear()
        with self.session_factory() as session:
            persons = session.scalars(select(Person).order_by(Person.name)).all()
            for person in persons:
                item = QtWidgets.QListWidgetItem(person.name)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if person.id in checked
                    else QtCore.Qt.CheckState.Unchecked
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, person.id)
                self.person_list_widget.addItem(item)
        enabled = self.person_list_widget.count() > 0
        self.person_list_widget.blockSignals(False)
        self.person_filter_btn.setEnabled(enabled)
        self._apply_search_filter(self.search_edit.text())
        self._update_button_text()

    def _apply_search_filter(self, text: str) -> None:
        lower = text.lower()
        for i in range(self.person_list_widget.count()):
            item = self.person_list_widget.item(i)
            if item is None:
                continue
            hidden = bool(lower) and lower not in item.text().lower()
            item.setHidden(hidden)

    def _on_search_changed(self, text: str) -> None:
        self._apply_search_filter(text)

    def _deselect_all(self) -> None:
        self.person_list_widget.blockSignals(True)
        for i in range(self.person_list_widget.count()):
            item = self.person_list_widget.item(i)
            if item is not None:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.person_list_widget.blockSignals(False)
        self._update_button_text()
        self.load_images()

    def _on_person_filter_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        self._update_button_text()
        self.load_images()

    def _update_button_text(self) -> None:
        n = len(self._selected_person_ids())
        if n == 0:
            self.person_filter_btn.setText(self.tr("Filter by Person"))
        else:
            self.person_filter_btn.setText(
                self.tr("{n} person(s) selected").format(n=n)
            )

    def _selected_person_ids(self) -> list[int]:
        ids = []
        for i in range(self.person_list_widget.count()):
            item = self.person_list_widget.item(i)
            if item and item.checkState() == QtCore.Qt.CheckState.Checked:
                ids.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return ids

    def _build_images_data(self, images: list) -> list:
        result = []
        for img in images:
            thumb_path = (
                self.paths.thumbs_dir / f"{img.file_hash}.jpg"
                if img.file_hash
                else self.paths.thumbs_dir / "unknown.jpg"
            )
            result.append((img.id, img.file_path, thumb_path))
        return result

    def load_images(self) -> None:
        person_ids = self._selected_person_ids()

        if not person_ids or self.vector_store is None:
            with self.session_factory() as session:
                stmt = select(Image)
                images = session.execute(stmt).unique().scalars().all()
                self.grid.set_results(self._build_images_data(images))
            return

        # Intersect FAISS results: only images where ALL selected persons appear
        per_person_scores: list[dict[int, float]] = []
        for person_id in person_ids:
            scores: dict[int, float] = {}
            for img_id, score in find_images_by_person(
                self.session_factory, self.vector_store, person_id
            ):
                scores[img_id] = score
            per_person_scores.append(scores)

        common_ids = set(per_person_scores[0].keys())
        for scores in per_person_scores[1:]:
            common_ids &= scores.keys()

        # Rank by minimum score across persons (weakest match determines relevance)
        image_scores: dict[int, float] = {
            img_id: min(s[img_id] for s in per_person_scores) for img_id in common_ids
        }

        if not image_scores:
            self.grid.set_results([])
            return

        sorted_pairs = sorted(image_scores.items(), key=lambda kv: kv[1], reverse=True)
        ordered_ids = [img_id for img_id, _ in sorted_pairs]

        with self.session_factory() as session:
            stmt = select(Image).where(Image.id.in_(ordered_ids))
            images = session.execute(stmt).unique().scalars().all()
            image_map = {img.id: img for img in images}
            ordered_images = [image_map[i] for i in ordered_ids if i in image_map]
            self.grid.set_results(self._build_images_data(ordered_images))

    def _on_navigate_to_labelling(self, image_id: int) -> None:
        from photoaident.app import MainWindow  # local import breaks circular dep

        main = self.window()
        if isinstance(main, MainWindow):
            main.go_to_labelling(image_id)
