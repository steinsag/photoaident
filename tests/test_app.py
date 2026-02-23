from photoaident.app import MainWindow
from photoaident.paths import AppPaths
from photoaident.db.migrate import apply_migrations


def _make_window(tmp_path, qtbot, collection_path: str = "") -> MainWindow:
    """Helper: create a MainWindow with migrations applied."""
    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")
    window = MainWindow(paths, check_gpu=False)
    window.settings.collection_path = collection_path
    qtbot.addWidget(window)
    return window


def test_app_setup(qtbot, tmp_path):
    """
    Test that the application window can be instantiated.
    """
    window = _make_window(tmp_path, qtbot)

    assert window.windowTitle() == "PhotoAIdent"
    assert window.library_page is not None
    assert window.labelling_page is not None


def test_startup_triggers_inventory_scan_when_collection_path_set(qtbot, tmp_path):
    """_maybe_start_indexing starts an InventoryTask when a collection path is set."""
    collection_dir = tmp_path / "photos"
    collection_dir.mkdir()

    window = _make_window(tmp_path, qtbot, collection_path=str(collection_dir))
    # Ensure any auto-timer-started task is cleaned up first
    if window._inventory_task is not None:
        window._inventory_task.cancel()
        if window._inventory_thread:
            window._inventory_thread.quit()
            window._inventory_thread.wait(3000)
        window._inventory_task = None
        window._inventory_thread = None
    if window._indexing_task is not None:
        window._indexing_task.cancel()
        if window._indexing_thread:
            window._indexing_thread.quit()
            window._indexing_thread.wait(3000)
        window._indexing_task = None
        window._indexing_thread = None

    window._maybe_start_indexing()

    assert window._inventory_task is not None

    # Cleanup
    window._inventory_task.cancel()
    if window._inventory_thread:
        window._inventory_thread.quit()
        window._inventory_thread.wait(3000)
    window._inventory_task = None
    window._inventory_thread = None


def test_startup_skips_scan_when_no_collection_path(qtbot, tmp_path):
    """_maybe_start_indexing does nothing when collection_path is empty."""
    window = _make_window(tmp_path, qtbot, collection_path="")
    # Reset any task started by the timer
    if window._inventory_task is not None:
        window._inventory_task.cancel()
        if window._inventory_thread:
            window._inventory_thread.quit()
            window._inventory_thread.wait(3000)
        window._inventory_task = None
        window._inventory_thread = None
    if window._indexing_task is not None:
        window._indexing_task.cancel()
        if window._indexing_thread:
            window._indexing_thread.quit()
            window._indexing_thread.wait(3000)
        window._indexing_task = None
        window._indexing_thread = None

    window._maybe_start_indexing()

    assert window._inventory_task is None
    assert window._indexing_task is None


def test_startup_scan_idempotent_when_already_running(qtbot, tmp_path):
    """Calling _maybe_start_indexing twice does not start a second inventory."""
    collection_dir = tmp_path / "photos"
    collection_dir.mkdir()

    window = _make_window(tmp_path, qtbot, collection_path=str(collection_dir))
    # Reset any timer-started tasks
    if window._inventory_task is not None:
        window._inventory_task.cancel()
        if window._inventory_thread:
            window._inventory_thread.quit()
            window._inventory_thread.wait(3000)
        window._inventory_task = None
        window._inventory_thread = None
    if window._indexing_task is not None:
        window._indexing_task.cancel()
        if window._indexing_thread:
            window._indexing_thread.quit()
            window._indexing_thread.wait(3000)
        window._indexing_task = None
        window._indexing_thread = None

    window._maybe_start_indexing()
    first_task = window._inventory_task
    assert first_task is not None

    # Second call must be a no-op
    window._maybe_start_indexing()
    assert window._inventory_task is first_task

    # Cleanup
    first_task.cancel()
    if window._inventory_thread:
        window._inventory_thread.quit()
        window._inventory_thread.wait(3000)
    window._inventory_task = None
    window._inventory_thread = None
