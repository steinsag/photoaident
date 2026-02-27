"""Tests for the Browse page (Miller columns folder navigation)."""

from PySide6 import QtCore

from photoaident.db.database import Image, get_engine, get_session_factory
from photoaident.db.migrate import apply_migrations
from photoaident.paths import AppPaths
from photoaident.settings import Settings
from photoaident.ui.pages.browse import BrowsePage


def _make_browse_page(tmp_path, qtbot, collection_path: str = "") -> BrowsePage:
    """Create a BrowsePage with isolated DB and optional collection path."""
    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")
    engine = get_engine(str(paths.db_path))
    session_factory = get_session_factory(engine)
    settings = Settings(collection_path=collection_path)
    page = BrowsePage(session_factory, paths, settings)
    qtbot.addWidget(page)
    return page


# ---------------------------------------------------------------------------
# 1. Smoke test
# ---------------------------------------------------------------------------


def test_browse_page_renders(tmp_path, qtbot):
    """BrowsePage instantiates without error and exposes a grid widget."""
    page = _make_browse_page(tmp_path, qtbot)
    assert page.grid is not None


# ---------------------------------------------------------------------------
# 2. Empty / missing collection path
# ---------------------------------------------------------------------------


def test_browse_page_empty_collection(tmp_path, qtbot):
    """Calling refresh() with no collection_path produces no columns and no crash."""
    page = _make_browse_page(tmp_path, qtbot, collection_path="")
    page.refresh()
    assert len(page._columns) == 0


# ---------------------------------------------------------------------------
# 3. Column 0 shows root; column 1 shows root's subfolders
# ---------------------------------------------------------------------------


def test_browse_page_root_column_shows_subfolders(tmp_path, qtbot):
    """After refresh(), column 0 holds the root folder and column 1 its children."""
    collection = tmp_path / "photos"
    collection.mkdir()
    (collection / "2020").mkdir()
    (collection / "2021").mkdir()

    page = _make_browse_page(tmp_path, qtbot, collection_path=str(collection))
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


def test_browse_page_root_images_shown_on_open(tmp_path, qtbot):
    """Images directly in the collection root are shown when the page opens."""
    collection = tmp_path / "photos"
    collection.mkdir()
    sub = collection / "sub"
    sub.mkdir()

    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")
    engine = get_engine(str(paths.db_path))
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        session.add(Image(file_path=str(collection / "root_img.jpg"), file_size=100))
        session.add(Image(file_path=str(sub / "sub_img.jpg"), file_size=100))
        session.commit()

    settings = Settings(collection_path=str(collection))
    page = BrowsePage(session_factory, paths, settings)
    qtbot.addWidget(page)
    page.refresh()

    # Root is auto-selected: only root_img.jpg should be in the grid
    assert page.grid._all_results is not None
    assert len(page.grid._all_results) == 1


# ---------------------------------------------------------------------------
# 5. Direct images loaded, subdirectory images excluded
# ---------------------------------------------------------------------------


def test_browse_page_loads_direct_images_only(tmp_path, qtbot):
    """Selecting a folder only loads images whose direct parent is that folder."""
    collection = tmp_path / "photos"
    collection.mkdir()
    sub_a = collection / "subA"
    sub_a.mkdir()
    sub_b = sub_a / "subB"
    sub_b.mkdir()

    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")
    engine = get_engine(str(paths.db_path))
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        session.add(Image(file_path=str(sub_a / "img1.jpg"), file_size=100))
        session.add(Image(file_path=str(sub_a / "img2.jpg"), file_size=100))
        session.add(Image(file_path=str(sub_b / "deep.jpg"), file_size=100))
        session.commit()

    settings = Settings(collection_path=str(collection))
    page = BrowsePage(session_factory, paths, settings)
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


def test_browse_page_column_hierarchy(tmp_path, qtbot):
    """Selecting a folder with subfolders adds a new column; a leaf removes it."""
    collection = tmp_path / "photos"
    collection.mkdir()
    parent_dir = collection / "events"
    parent_dir.mkdir()
    (parent_dir / "wedding").mkdir()
    (parent_dir / "birthday").mkdir()
    leaf_dir = collection / "misc"
    leaf_dir.mkdir()

    page = _make_browse_page(tmp_path, qtbot, collection_path=str(collection))
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


def test_browse_page_column_replaced_on_new_selection(tmp_path, qtbot):
    """Selecting a new folder replaces child columns to the right."""
    collection = tmp_path / "photos"
    collection.mkdir()
    dir_a = collection / "dirA"
    dir_a.mkdir()
    (dir_a / "child1").mkdir()

    dir_b = collection / "dirB"
    dir_b.mkdir()
    (dir_b / "other").mkdir()

    page = _make_browse_page(tmp_path, qtbot, collection_path=str(collection))
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


def test_browse_page_reclick_ancestor_collapses_child_columns(tmp_path, qtbot):
    """Re-clicking col 0 (root) while col 1 and col 2 are open removes col 2."""
    collection = tmp_path / "photos"
    collection.mkdir()
    sub = collection / "sub"
    sub.mkdir()
    (sub / "grand").mkdir()

    page = _make_browse_page(tmp_path, qtbot, collection_path=str(collection))
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
