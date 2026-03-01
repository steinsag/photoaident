import bisect
from typing import Optional

from PySide6 import QtCore, QtWidgets

from photoaident.db.database import Person


class PersonListWidget(QtWidgets.QWidget):
    """Left panel of the person selector: search field, list, and New Person button."""

    person_selected = QtCore.Signal(object)  # emits Optional[Person]
    new_person_requested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(250)
        self._all_persons: list[Person] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText(self.tr("Type to filter"))
        self._search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search_edit)

        self._list_widget = QtWidgets.QListWidget()
        self._list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._list_widget.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list_widget, stretch=1)

        self._new_person_btn = QtWidgets.QPushButton(self.tr("New Person\u2026"))
        self._new_person_btn.clicked.connect(self.new_person_requested)
        layout.addWidget(self._new_person_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_persons(self, persons: list[Person]) -> None:
        """Replace the full person list and re-apply the current filter."""
        self._all_persons = list(persons)
        self._apply_filter(self._search_edit.text())

    def add_person_sorted(self, person: Person) -> None:
        """Insert person in sorted order and re-apply the current filter."""
        bisect.insort(self._all_persons, person, key=lambda p: p.name)
        self._apply_filter(self._search_edit.text())

    def select_by_id(self, person_id: int) -> bool:
        """Select the list item whose Person has the given id.

        Returns True on success.
        """
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item is None:  # pragma: no cover
                continue  # pragma: no cover
            if item.data(QtCore.Qt.ItemDataRole.UserRole).id == person_id:
                self._list_widget.setCurrentItem(item)
                return True
        return False

    def clear_selection(self) -> None:
        """Deselect all items and emit person_selected(None)."""
        self._list_widget.setCurrentRow(-1)

    def current_filter_text(self) -> str:
        """Return the current search field text."""
        return self._search_edit.text()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _apply_filter(self, text: str) -> None:
        needle = text.lower().strip()
        filtered = (
            [p for p in self._all_persons if needle in p.name.lower()]
            if needle
            else self._all_persons
        )
        self._list_widget.clear()
        for person in filtered:
            item = QtWidgets.QListWidgetItem(person.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, person)
            self._list_widget.addItem(item)

    def _on_item_changed(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        _: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        if current is None:
            self.person_selected.emit(None)
        else:
            self.person_selected.emit(current.data(QtCore.Qt.ItemDataRole.UserRole))
