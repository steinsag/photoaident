from PySide6 import QtCore

from photoaident.db.database import Person
from photoaident.ui.widgets.person_list_widget import PersonListWidget


def _make_persons(*names: str) -> list[Person]:
    """Return a list of unsaved Person objects with the given names."""
    persons = []
    for i, name in enumerate(names, start=1):
        p = Person(name=name)
        p.id = i
        persons.append(p)
    return persons


# ===========================================================================
# set_persons / filtering
# ===========================================================================


def test_set_persons_populates_list(qtbot):
    """set_persons() fills the list widget with the given persons."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    widget.set_persons(_make_persons("Alice", "Bob"))

    assert widget._list_widget.count() == 2
    assert widget._list_widget.item(0).text() == "Alice"
    assert widget._list_widget.item(1).text() == "Bob"


def test_set_persons_replaces_existing(qtbot):
    """Calling set_persons() again replaces the previous list."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    widget.set_persons(_make_persons("Alice", "Bob"))
    widget.set_persons(_make_persons("Carol"))

    assert widget._list_widget.count() == 1
    assert widget._list_widget.item(0).text() == "Carol"


def test_filter_with_text_narrows_list(qtbot):
    """_apply_filter() shows only persons whose names contain the search text."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    widget.set_persons(_make_persons("Alice", "Bob"))

    widget._apply_filter("ali")

    assert widget._list_widget.count() == 1
    assert widget._list_widget.item(0).text() == "Alice"


def test_filter_empty_text_shows_all(qtbot):
    """_apply_filter() with empty text shows the full list."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    widget.set_persons(_make_persons("Alice", "Bob"))

    widget._apply_filter("ali")
    assert widget._list_widget.count() == 1

    widget._apply_filter("")
    assert widget._list_widget.count() == 2


def test_filter_is_case_insensitive(qtbot):
    """_apply_filter() matches regardless of case."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    widget.set_persons(_make_persons("Alice"))

    widget._apply_filter("ALICE")
    assert widget._list_widget.count() == 1


# ===========================================================================
# add_person_sorted
# ===========================================================================


def test_add_person_sorted_inserts_in_alphabetical_order(qtbot):
    """add_person_sorted() inserts a person at the correct sorted position."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    widget.set_persons(_make_persons("Alice", "Carol"))

    bob = Person(name="Bob")
    bob.id = 99
    widget.add_person_sorted(bob)

    assert widget._list_widget.count() == 3
    texts = [widget._list_widget.item(i).text() for i in range(3)]
    assert texts == ["Alice", "Bob", "Carol"]


# ===========================================================================
# select_by_id / clear_selection
# ===========================================================================


def test_select_by_id_selects_matching_item(qtbot):
    """select_by_id() highlights the item with the matching person id."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    persons = _make_persons("Alice", "Bob")
    widget.set_persons(persons)

    found = widget.select_by_id(persons[1].id)  # Bob

    assert found is True
    assert widget._list_widget.currentItem().text() == "Bob"


def test_select_by_id_returns_false_when_not_found(qtbot):
    """select_by_id() returns False when no item has the given id."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    widget.set_persons(_make_persons("Alice"))

    found = widget.select_by_id(999)

    assert found is False


def test_clear_selection_deselects_current_item(qtbot):
    """clear_selection() deselects without removing items."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    persons = _make_persons("Alice")
    widget.set_persons(persons)
    widget.select_by_id(persons[0].id)
    assert widget._list_widget.currentRow() == 0

    widget.clear_selection()

    assert widget._list_widget.currentRow() == -1


# ===========================================================================
# current_filter_text
# ===========================================================================


def test_current_filter_text_returns_search_field_value(qtbot):
    """current_filter_text() returns whatever is typed in the search field."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    widget._search_edit.setText("test query")

    assert widget.current_filter_text() == "test query"


# ===========================================================================
# Signals
# ===========================================================================


def test_person_selected_signal_emitted_on_item_click(qtbot):
    """person_selected emits the Person when an item is selected."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    persons = _make_persons("Alice")
    widget.set_persons(persons)

    received: list = []
    widget.person_selected.connect(received.append)

    widget._list_widget.setCurrentRow(0)

    assert len(received) == 1
    assert received[0] is not None
    assert received[0].name == "Alice"


def test_person_selected_signal_emits_none_on_deselect(qtbot):
    """person_selected emits None when the selection is cleared."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    persons = _make_persons("Alice")
    widget.set_persons(persons)
    widget._list_widget.setCurrentRow(0)

    received: list = []
    widget.person_selected.connect(received.append)

    widget.clear_selection()

    assert len(received) == 1
    assert received[0] is None


def test_new_person_requested_signal_emitted_on_button_click(qtbot):
    """new_person_requested is emitted when the New Person… button is clicked."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.new_person_requested):
        widget._new_person_btn.click()


def test_item_stores_person_in_user_role(qtbot):
    """Each list item stores its Person object in ItemDataRole.UserRole."""
    widget = PersonListWidget()
    qtbot.addWidget(widget)
    persons = _make_persons("Alice", "Bob")
    widget.set_persons(persons)

    item = widget._list_widget.item(0)
    stored: Person = item.data(QtCore.Qt.ItemDataRole.UserRole)

    assert stored.name == "Alice"
    assert stored.id == persons[0].id
