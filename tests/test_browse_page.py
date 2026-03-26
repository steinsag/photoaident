"""Tests for the Browse page (Miller columns folder navigation)."""

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.db.database import Image, get_engine, get_session_factory
from photoaident.db.migrate import apply_migrations
from photoaident.db.vector_store import VectorStore
from photoaident.paths import AppPaths
from photoaident.settings import Settings
from photoaident.ui.pages.browse import BrowsePage


def _make_browse_page(
    tmp_app_paths: AppPaths, qtbot, collection_path: str = ""
) -> BrowsePage:
    """Create a BrowsePage with isolated DB and optional collection path."""
    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")
    engine = get_engine(str(tmp_app_paths.db_path))
    session_factory = get_session_factory(engine)
    settings = Settings(collection_path=collection_path)
    vector_store = VectorStore()
    page = BrowsePage(session_factory, tmp_app_paths, settings, vector_store)
    qtbot.addWidget(page)
    return page


# ---------------------------------------------------------------------------
# 1. Smoke test
# ---------------------------------------------------------------------------


def test_browse_page_renders(tmp_app_paths, qtbot):
    """BrowsePage instantiates without error and exposes a grid widget."""
    page = _make_browse_page(tmp_app_paths, qtbot)
    assert page.grid is not None


# ---------------------------------------------------------------------------
# 2. Empty / missing collection path
# ---------------------------------------------------------------------------


def test_browse_page_empty_collection(tmp_app_paths, qtbot):
    """Calling refresh() with no collection_path produces no columns and no crash."""
    page = _make_browse_page(tmp_app_paths, qtbot, collection_path="")
    page.refresh()
    assert len(page._columns) == 0


# ---------------------------------------------------------------------------
# 3. Column 0 shows root; column 1 shows root's subfolders
# ---------------------------------------------------------------------------


def test_browse_page_root_column_shows_subfolders(tmp_app_paths, qtbot):
    """After refresh(), column 0 holds the root folder and column 1 its children."""
    collection = tmp_app_paths.thumbs_dir / "photos"
    collection.mkdir()
    (collection / "2020").mkdir()
    (collection / "2021").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    # Column 0: root itself (single item, auto-selected)
    assert len(page._columns) >= 1
    assert page._columns[0].count() == 1
    assert page._columns[0].item(0).text() == "photos"

    # Column 1: root's subfolders
    assert len(page._columns) == 2
    assert page._columns[1].count() == 2
    names = {page._columns[1].item(i).text() for i in range(page._columns[1].count())}
    assert names == {"2020", "2021"}


# ---------------------------------------------------------------------------
# 4. Root-level images shown on open
# ---------------------------------------------------------------------------


def test_browse_page_root_images_shown_on_open(tmp_app_paths, qtbot):
    """Images directly in the collection root are shown when the page opens."""
    collection = _make_collection_dir(tmp_app_paths)

    sub = collection / "sub"
    sub.mkdir()

    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")
    engine = get_engine(str(tmp_app_paths.db_path))
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        session.add(Image(file_path=str(collection / "root_img.jpg"), file_size=100))
        session.add(Image(file_path=str(sub / "sub_img.jpg"), file_size=100))
        session.commit()

    settings = Settings(collection_path=str(collection))
    vector_store = VectorStore()
    page = BrowsePage(session_factory, tmp_app_paths, settings, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    # Root is auto-selected: only root_img.jpg should be in the grid
    assert page.grid._all_results is not None
    assert len(page.grid._all_results) == 1


# ---------------------------------------------------------------------------
# 5. Direct images loaded, subdirectory images excluded
# ---------------------------------------------------------------------------


def test_browse_page_loads_direct_images_only(tmp_app_paths, qtbot):
    """Selecting a folder only loads images whose direct parent is that folder."""
    collection = _make_collection_dir(tmp_app_paths)

    sub_a = collection / "subA"
    sub_a.mkdir()
    sub_b = sub_a / "subB"
    sub_b.mkdir()

    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")
    engine = get_engine(str(tmp_app_paths.db_path))
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        session.add(Image(file_path=str(sub_a / "img1.jpg"), file_size=100))
        session.add(Image(file_path=str(sub_a / "img2.jpg"), file_size=100))
        session.add(Image(file_path=str(sub_b / "deep.jpg"), file_size=100))
        session.commit()

    settings = Settings(collection_path=str(collection))
    vector_store = VectorStore()
    page = BrowsePage(session_factory, tmp_app_paths, settings, vector_store)
    qtbot.addWidget(page)
    page.refresh()
    # After refresh: col 0 = root (auto-selected), col 1 = ["subA"]

    # Select "subA" in column 1 by emitting itemClicked (mirrors a mouse click)
    col1 = page._columns[1]
    for i in range(col1.count()):
        if col1.item(i).text() == "subA":
            col1.itemClicked.emit(col1.item(i))
            break

    QtCore.QCoreApplication.processEvents()

    # Only the 2 direct children of subA should be in the grid
    assert page.grid._all_results is not None
    assert len(page.grid._all_results) == 2


# ---------------------------------------------------------------------------
# 6. Column hierarchy: selecting subfolder-bearing folder adds another column
# ---------------------------------------------------------------------------


def test_browse_page_column_hierarchy(tmp_app_paths, qtbot):
    """Selecting a folder with subfolders adds a new column; a leaf removes it."""
    collection = _make_collection_dir(tmp_app_paths)

    parent_dir = collection / "events"
    parent_dir.mkdir()
    (parent_dir / "wedding").mkdir()
    (parent_dir / "birthday").mkdir()
    leaf_dir = collection / "misc"
    leaf_dir.mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()
    # col 0 = root (auto-selected), col 1 = ["events", "misc"]
    assert len(page._columns) == 2

    col1 = page._columns[1]

    # Select "events" (has children) → column 2 appears
    for i in range(col1.count()):
        if col1.item(i).text() == "events":
            col1.itemClicked.emit(col1.item(i))
            break
    QtCore.QCoreApplication.processEvents()
    assert len(page._columns) == 3
    assert page._columns[2].count() == 2

    # Select "misc" (leaf, no children) → column 2 removed
    for i in range(col1.count()):
        if col1.item(i).text() == "misc":
            col1.itemClicked.emit(col1.item(i))
            break
    QtCore.QCoreApplication.processEvents()
    assert len(page._columns) == 2


# ---------------------------------------------------------------------------
# 7. Column replaced when a different folder is selected in the same column
# ---------------------------------------------------------------------------


def test_browse_page_column_replaced_on_new_selection(tmp_app_paths, qtbot):
    """Selecting a new folder replaces child columns to the right."""
    collection = _make_collection_dir(tmp_app_paths)

    dir_a = collection / "dirA"
    dir_a.mkdir()
    (dir_a / "child1").mkdir()

    dir_b = collection / "dirB"
    dir_b.mkdir()
    (dir_b / "other").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()
    # col 0 = root (auto-selected), col 1 = ["dirA", "dirB"]

    col1 = page._columns[1]

    # Select dirA → column 2 appears with "child1"
    for i in range(col1.count()):
        if col1.item(i).text() == "dirA":
            col1.itemClicked.emit(col1.item(i))
            break
    QtCore.QCoreApplication.processEvents()
    assert len(page._columns) == 3
    first_col2_item = page._columns[2].item(0)
    assert first_col2_item is not None
    assert first_col2_item.text() == "child1"

    # Select dirB → column 2 replaced with "other"
    for i in range(col1.count()):
        if col1.item(i).text() == "dirB":
            col1.itemClicked.emit(col1.item(i))
            break
    QtCore.QCoreApplication.processEvents()
    assert len(page._columns) == 3
    new_col2_item = page._columns[2].item(0)
    assert new_col2_item is not None
    assert new_col2_item.text() == "other"


# ---------------------------------------------------------------------------
# 8. Re-clicking an already-selected ancestor collapses child columns (the
#    Finder re-click bug that currentItemChanged cannot catch)
# ---------------------------------------------------------------------------


def test_browse_page_reclick_ancestor_collapses_child_columns(tmp_app_paths, qtbot):
    """Re-clicking col 0 (root) while col 1 and col 2 are open removes col 2."""
    collection = _make_collection_dir(tmp_app_paths)

    sub = collection / "sub"
    sub.mkdir()
    (sub / "grand").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()
    # col 0 = root (auto-selected), col 1 = ["sub"]

    # Navigate into "sub" → col 2 = ["grand"]
    col1 = page._columns[1]
    col1.itemClicked.emit(col1.item(0))
    QtCore.QCoreApplication.processEvents()
    assert len(page._columns) == 3

    # Re-click root in col 0 (already selected) → col 2 must be removed
    col0 = page._columns[0]
    col0.itemClicked.emit(col0.item(0))
    QtCore.QCoreApplication.processEvents()
    assert len(page._columns) == 2  # col 0 + col 1 (sub's children)


# ---------------------------------------------------------------------------
# 9. Non-existent collection path clears UI and shows hint (lines 91–94)
# ---------------------------------------------------------------------------


def test_browse_page_nonexistent_collection(tmp_app_paths, qtbot):
    """refresh() with a configured-but-missing directory clears UI and shows hint."""
    collection = _make_collection_dir(tmp_app_paths)

    page = _make_browse_page(
        tmp_app_paths, qtbot, collection_path=str(collection / "gone")
    )
    page.refresh()
    assert len(page._columns) == 0
    assert not page._hint_label.isHidden()


# ---------------------------------------------------------------------------
# 10. _clear_columns loop body exercises deletion of existing columns (lines 106–107)
# ---------------------------------------------------------------------------


def test_browse_page_clear_columns_removes_existing(tmp_app_paths, qtbot):
    """Changing collection root exercises the deletion loop inside _clear_columns."""
    collection = _make_collection_dir(tmp_app_paths)
    (collection / "sub").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()
    assert len(page._columns) == 2

    # Change root to force a full rebuild (same root would short-circuit)
    new_collection = tmp_app_paths.thumbs_dir / "other_photos"
    new_collection.mkdir()
    (new_collection / "child").mkdir()
    page.settings.collection_path = str(new_collection)
    page.refresh()
    assert len(page._columns) == 2


# ---------------------------------------------------------------------------
# 11. None-guard in _on_column_item_changed is a no-op (line 182)
# ---------------------------------------------------------------------------


def test_browse_page_column_item_changed_none_is_noop(tmp_app_paths, qtbot):
    """Passing None to _on_column_item_changed must not raise or mutate state."""
    collection = _make_collection_dir(tmp_app_paths)

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()
    before = len(page._columns)
    page._on_column_item_changed(0, None)  # must be a no-op
    assert len(page._columns) == before


# ---------------------------------------------------------------------------
# 12. PermissionError in _get_subfolders returns [] (lines 207–208)
# ---------------------------------------------------------------------------


def test_browse_page_permission_error_returns_empty(qtbot, monkeypatch, tmp_app_paths):
    """_get_subfolders returns [] when the OS raises PermissionError."""
    page = _make_browse_page(tmp_app_paths, qtbot)
    collection = _make_collection_dir(tmp_app_paths)

    original_iterdir = Path.iterdir

    def raise_permission(self):
        if self == collection:
            raise PermissionError("access denied")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", raise_permission)
    assert page._get_subfolders(collection) == []


# ---------------------------------------------------------------------------
# 13. navigate_to_labelling signal is emitted when grid requests navigation
# ---------------------------------------------------------------------------


def test_browse_page_navigate_to_labelling_signal(qtbot, tmp_app_paths):
    """BrowsePage.navigate_to_labelling signal is emitted when the grid emits it."""
    page = _make_browse_page(tmp_app_paths, qtbot)

    with qtbot.waitSignal(page.navigate_to_labelling, timeout=1000) as blocker:
        page.grid.navigate_to_labelling.emit(42)

    assert blocker.args == [42]


# ===========================================================================
# eventFilter — non-KeyPress events fall through to super
# ===========================================================================


def test_event_filter_non_keypress_falls_through(tmp_app_paths, qtbot):
    """eventFilter with a non-KeyPress event returns super()'s result."""
    collection = _make_collection_dir(tmp_app_paths)

    (collection / "sub").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    col = page._columns[0]
    # Send a mouse-button-press event (not a key event) directly to the filter.
    mouse_event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(0, 0),
        QtCore.QPointF(0, 0),
        QtCore.QPointF(0, 0),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    # super().eventFilter returns False for unhandled events; verify no crash
    # and that the return value is False (unhandled).
    result = page.eventFilter(col, mouse_event)
    assert result is False


# ===========================================================================
# eventFilter — KeyPress on a non-column object falls through to super
# ===========================================================================


def test_event_filter_keypress_on_non_column_falls_through(tmp_app_paths, qtbot):
    """eventFilter ignores KeyPress events from objects that are not columns."""
    collection = _make_collection_dir(tmp_app_paths)

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    unrelated = QtWidgets.QListWidget()
    qtbot.addWidget(unrelated)

    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Left,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    result = page.eventFilter(unrelated, key_event)
    assert result is False


# ===========================================================================
# eventFilter — Key_Left dispatches to _navigate_left
# ===========================================================================


def test_event_filter_key_left_calls_navigate_left(qtbot, monkeypatch, tmp_app_paths):
    """eventFilter on Key_Left calls _navigate_left with the column index."""
    collection = _make_collection_dir(tmp_app_paths)

    (collection / "sub").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    calls: list[int] = []
    monkeypatch.setattr(page, "_navigate_left", lambda idx: calls.append(idx))

    # Column 1 exists because root has a subfolder.
    col = page._columns[1]
    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Left,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    result = page.eventFilter(col, key_event)

    assert result is True
    assert calls == [1]


# ===========================================================================
# eventFilter — Key_Right dispatches to _navigate_right
# ===========================================================================


def test_event_filter_key_right_calls_navigate_right(qtbot, monkeypatch, tmp_app_paths):
    """eventFilter on Key_Right calls _navigate_right with the column index."""
    collection = _make_collection_dir(tmp_app_paths)

    (collection / "sub").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    calls: list[int] = []
    monkeypatch.setattr(page, "_navigate_right", lambda idx: calls.append(idx))

    # Use column 0 so there is something to the right.
    col = page._columns[0]
    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Right,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    result = page.eventFilter(col, key_event)

    assert result is True
    assert calls == [0]


# ===========================================================================
# eventFilter — Key_Down changes row and triggers _on_column_item_changed
# ===========================================================================


def test_event_filter_key_down_changes_row_triggers_handler(
    qtbot, monkeypatch, tmp_app_paths
):
    """Key_Down moves selection and calls _on_column_item_changed when row changes."""
    collection = _make_collection_dir(tmp_app_paths)

    (collection / "alpha").mkdir()
    (collection / "beta").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    # Column 1 has two subfolders; select the first one.
    col = page._columns[1]
    col.setCurrentRow(0)

    handler_calls: list[tuple[int, object]] = []
    monkeypatch.setattr(
        page,
        "_on_column_item_changed",
        lambda idx, item: handler_calls.append((idx, item)),
    )

    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Down,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    result = page.eventFilter(col, key_event)

    assert result is True
    # Row should have moved from 0 to 1, triggering the handler.
    assert col.currentRow() == 1
    assert len(handler_calls) == 1
    assert handler_calls[0][0] == 1  # col_index for column 1


# ===========================================================================
# eventFilter — Key_Up on first row does not trigger _on_column_item_changed
# ===========================================================================


def test_event_filter_key_up_at_first_row_no_handler_call(
    qtbot, monkeypatch, tmp_app_paths
):
    """Key_Up when already on row 0 does not call _on_column_item_changed."""
    collection = _make_collection_dir(tmp_app_paths)

    (collection / "sub").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    col = page._columns[1]
    col.setCurrentRow(0)

    handler_calls: list = []
    monkeypatch.setattr(
        page,
        "_on_column_item_changed",
        lambda idx, item: handler_calls.append((idx, item)),
    )

    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Up,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    result = page.eventFilter(col, key_event)

    assert result is True
    # Row did not change (already at top), so handler must not be called.
    assert handler_calls == []


# ===========================================================================
# eventFilter — unrecognised key falls through to super
# ===========================================================================


def test_event_filter_unrecognised_key_falls_through(tmp_app_paths, qtbot):
    """eventFilter returns super()'s result for unrecognised keys like Key_Space."""
    collection = _make_collection_dir(tmp_app_paths)

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    col = page._columns[0]
    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Space,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    result = page.eventFilter(col, key_event)
    # super().eventFilter returns False for unhandled events.
    assert result is False


# ===========================================================================
# _navigate_left — at column 0 is a no-op
# ===========================================================================


def test_navigate_left_at_column_zero_is_noop(qtbot, monkeypatch, tmp_app_paths):
    """_navigate_left(0) does nothing — there is no column to the left."""
    collection = _make_collection_dir(tmp_app_paths)

    (collection / "sub").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    handler_calls: list = []
    monkeypatch.setattr(
        page,
        "_on_column_item_changed",
        lambda idx, item: handler_calls.append((idx, item)),
    )

    page._navigate_left(0)

    assert handler_calls == []


# ===========================================================================
# _navigate_left — moves focus and calls _on_column_item_changed
# ===========================================================================


def test_navigate_left_focuses_previous_column(qtbot, monkeypatch, tmp_app_paths):
    """_navigate_left(1) focuses column 0 and fires _on_column_item_changed(0, ...)."""
    collection = _make_collection_dir(tmp_app_paths)

    (collection / "sub").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.show()
    page.refresh()

    handler_calls: list[tuple[int, object]] = []
    monkeypatch.setattr(
        page,
        "_on_column_item_changed",
        lambda idx, item: handler_calls.append((idx, item)),
    )

    page._navigate_left(1)

    assert len(handler_calls) == 1
    assert handler_calls[0][0] == 0  # col_index 0


# ===========================================================================
# _navigate_right — at last column is a no-op
# ===========================================================================


def test_navigate_right_at_last_column_is_noop(qtbot, monkeypatch, tmp_app_paths):
    """_navigate_right at the last column does nothing."""
    collection = _make_collection_dir(tmp_app_paths)

    # No subfolders under root: after refresh only col 0 exists.

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    handler_calls: list = []
    monkeypatch.setattr(
        page,
        "_on_column_item_changed",
        lambda idx, item: handler_calls.append((idx, item)),
    )

    last_col_index = len(page._columns) - 1
    page._navigate_right(last_col_index)

    assert handler_calls == []


# ===========================================================================
# _navigate_right — focuses right column, selects row 0 when nothing selected
# ===========================================================================


def test_navigate_right_selects_first_row_when_nothing_selected(
    qtbot, monkeypatch, tmp_app_paths
):
    """_navigate_right selects row 0 in the right column when it has no selection."""
    collection = _make_collection_dir(tmp_app_paths)

    (collection / "alpha").mkdir()
    (collection / "beta").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    # Clear any existing selection in column 1 so we exercise the setCurrentRow(0) path.
    page._columns[1].clearSelection()
    page._columns[1].setCurrentRow(-1)

    handler_calls: list[tuple[int, object]] = []
    monkeypatch.setattr(
        page,
        "_on_column_item_changed",
        lambda idx, item: handler_calls.append((idx, item)),
    )

    page._navigate_right(0)

    assert page._columns[1].currentRow() == 0
    assert len(handler_calls) == 1
    assert handler_calls[0][0] == 1  # col_index 1


# ===========================================================================
# _navigate_right — preserves existing selection in the right column
# ===========================================================================


def test_navigate_right_keeps_existing_selection(qtbot, monkeypatch, tmp_app_paths):
    """_navigate_right keeps the current item when the right column already has one."""
    collection = _make_collection_dir(tmp_app_paths)

    (collection / "alpha").mkdir()
    (collection / "beta").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    # Pre-select row 1 (second subfolder) in column 1.
    page._columns[1].setCurrentRow(1)

    handler_calls: list[tuple[int, object]] = []
    monkeypatch.setattr(
        page,
        "_on_column_item_changed",
        lambda idx, item: handler_calls.append((idx, item)),
    )

    page._navigate_right(0)

    # Selection should remain on row 1 (not reset to 0).
    assert page._columns[1].currentRow() == 1
    assert len(handler_calls) == 1
    assert handler_calls[0][0] == 1


# ===========================================================================
# _navigate_left — collapses child columns to the right
# ===========================================================================


def test_navigate_left_collapses_right_columns(tmp_app_paths, qtbot):
    """Pressing Key_Left in col 1 collapses any columns beyond col 0."""
    collection = _make_collection_dir(tmp_app_paths)

    sub = collection / "sub"
    sub.mkdir()
    (sub / "grand").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()

    # Navigate into "sub" to reveal col 2 with "grand".
    col1 = page._columns[1]
    col1.itemClicked.emit(col1.item(0))
    QtCore.QCoreApplication.processEvents()
    assert len(page._columns) == 3

    # Fire Key_Left on column 1 via eventFilter to collapse col 2.
    col1 = page._columns[1]
    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Left,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    page.eventFilter(col1, key_event)
    QtCore.QCoreApplication.processEvents()

    # After navigating left to col 0, _on_column_item_changed re-selects root,
    # which re-adds col 1 (sub's children) but col 2 is gone.
    assert len(page._columns) == 2


# ===========================================================================
# _navigate_right — expanding into subfolder adds a new column
# ===========================================================================


def test_navigate_right_expands_into_subfolder(tmp_app_paths, qtbot):
    """Pressing Key_Right in col 0 expands root's subfolder and adds col 2."""
    collection = _make_collection_dir(tmp_app_paths)

    sub = collection / "sub"
    sub.mkdir()
    (sub / "grand").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()
    # After refresh: col 0 = root, col 1 = ["sub"].  Ensure col 1 row 0 is selected.
    page._columns[1].setCurrentRow(0)
    QtCore.QCoreApplication.processEvents()

    # Now navigate right from col 1 into col 2 (grand's children or lack thereof).
    # But col 1 item "sub" has subfolder "grand", so pressing right on col 0
    # should work too.  Use col 0 → moves to col 1 item "sub".
    key_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Right,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    page.eventFilter(page._columns[0], key_event)
    QtCore.QCoreApplication.processEvents()

    # "sub" has a subfolder, so a new column should appear as col 2 with "grand".
    assert len(page._columns) == 3
    grand_items = [
        page._columns[2].item(row).text() for row in range(page._columns[2].count())
    ]
    assert "grand" in grand_items


# ===========================================================================
# refresh() preserves folder selection across page switches
# ===========================================================================


def test_refresh_preserves_folder_selection(tmp_app_paths, qtbot):
    """Second refresh() with same root preserves columns and grid results."""
    collection = _make_collection_dir(tmp_app_paths)

    sub = collection / "sub"
    sub.mkdir()

    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")
    engine = get_engine(str(tmp_app_paths.db_path))
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        session.add(Image(file_path=str(sub / "img.jpg"), file_size=100))
        session.commit()

    settings = Settings(collection_path=str(collection))
    vector_store = VectorStore()
    page = BrowsePage(session_factory, tmp_app_paths, settings, vector_store)
    qtbot.addWidget(page)
    page.refresh()

    # Navigate into "sub"
    col1 = page._columns[1]
    col1.itemClicked.emit(col1.item(0))
    QtCore.QCoreApplication.processEvents()

    columns_before = len(page._columns)
    grid_results_before = len(page.grid._all_results or [])
    selected_before = page._selected_path

    # Simulate switching away and back
    page.refresh()

    assert len(page._columns) == columns_before
    assert len(page.grid._all_results or []) == grid_results_before
    assert page._selected_path == selected_before


# ===========================================================================
# refresh() rebuilds when collection path changes
# ===========================================================================


def test_refresh_rebuilds_when_collection_path_changes(tmp_app_paths, qtbot):
    """Changing collection_path between refreshes rebuilds columns from new root."""
    collection1 = _make_collection_dir(tmp_app_paths)
    (collection1 / "folder_a").mkdir()

    collection2 = tmp_app_paths.thumbs_dir / "other"
    collection2.mkdir()
    (collection2 / "folder_x").mkdir()

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection1))
    page.refresh()

    # Column 1 should have "folder_a"
    assert page._columns[1].item(0).text() == "folder_a"

    # Switch collection path
    page.settings.collection_path = str(collection2)
    page.refresh()

    # Column 1 should now have "folder_x"
    assert page._columns[1].item(0).text() == "folder_x"
    assert page._current_root == collection2


# ===========================================================================
# refresh() resets _current_root when collection is cleared
# ===========================================================================


def test_refresh_resets_root_when_collection_cleared(tmp_app_paths, qtbot):
    """Clearing collection_path resets _current_root to None."""
    collection = _make_collection_dir(tmp_app_paths)

    page = _make_browse_page(tmp_app_paths, qtbot, collection_path=str(collection))
    page.refresh()
    assert page._current_root is not None

    page.settings.collection_path = ""
    page.refresh()
    assert page._current_root is None
    assert len(page._columns) == 0


def _make_collection_dir(tmp_app_paths: AppPaths) -> Path:
    collection_dir = tmp_app_paths.thumbs_dir / "photos"
    collection_dir.mkdir(exist_ok=True)
    return collection_dir
