from photoaident.app import MainWindow
from photoaident.paths import AppPaths
from photoaident.db.migrate import apply_migrations
from photoaident.db.database import Image, Face
from photoaident.settings import Settings


def _make_window(tmp_path, qtbot, collection_path: str = "") -> MainWindow:
    """Helper: create a MainWindow with migrations applied. Onboarding is disabled."""
    paths = AppPaths(
        base_data=tmp_path / "data",
        base_cache=tmp_path / "cache",
        base_config=tmp_path / "config",
    )
    paths.ensure_dirs()
    apply_migrations(f"sqlite:///{paths.db_path}")
    window = MainWindow(paths, check_gpu=False, enable_onboarding=False)
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


def test_counts_label_shows_zeros_on_empty_db(qtbot, tmp_path):
    """counts_label shows 0 images and 0 faces when the database is empty."""
    window = _make_window(tmp_path, qtbot)
    text = window.counts_label.text()
    assert "0" in text


def test_counts_label_reflects_db_contents(qtbot, tmp_path):
    """counts_label shows the correct totals loaded from the database."""
    window = _make_window(tmp_path, qtbot)

    # Insert one image and two faces directly via the session factory
    with window.session_factory() as session:
        img = Image(file_path="/test/photo.jpg", file_hash="abc123", file_size=1000)
        session.add(img)
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                faiss_id=0,
                bbox_x=0,
                bbox_y=0,
                bbox_w=10,
                bbox_h=10,
                detection_confidence=0.99,
                model_version="v1",
            )
        )
        session.add(
            Face(
                image_id=img.id,
                faiss_id=1,
                bbox_x=20,
                bbox_y=0,
                bbox_w=10,
                bbox_h=10,
                detection_confidence=0.95,
                model_version="v1",
            )
        )
        session.commit()

    window._update_db_counts()

    text = window.counts_label.text()
    assert "1" in text  # 1 image
    assert "2" in text  # 2 faces


def test_counts_label_updated_after_indexing_finished(qtbot, tmp_path):
    """_on_indexing_finished refreshes the counts label."""
    window = _make_window(tmp_path, qtbot)

    with window.session_factory() as session:
        img = Image(file_path="/test/photo2.jpg", file_hash="def456", file_size=500)
        session.add(img)
        session.commit()

    window._on_indexing_finished()

    text = window.counts_label.text()
    assert "1" in text  # 1 image


def test_onboarding_triggered_when_no_path(qtbot, tmp_path, monkeypatch):
    """_maybe_start_indexing triggers onboarding when no collection path is set."""
    window = _make_window(tmp_path, qtbot)
    # Use monkeypatch so the flag is restored to False before the 1-second timer fires
    monkeypatch.setattr(window, "_onboarding_enabled", True)

    onboarding_called: list[bool] = []
    monkeypatch.setattr(
        window, "_show_onboarding", lambda: onboarding_called.append(True)
    )

    window._maybe_start_indexing()

    assert len(onboarding_called) == 1
    assert window._inventory_task is None
    assert window._indexing_task is None


def test_onboarding_not_triggered_when_path_set(qtbot, tmp_path, monkeypatch):
    """_maybe_start_indexing does not trigger onboarding when a path is already set."""
    collection_dir = tmp_path / "photos"
    collection_dir.mkdir()
    window = _make_window(tmp_path, qtbot, collection_path=str(collection_dir))
    monkeypatch.setattr(window, "_onboarding_enabled", True)

    # Ensure no background tasks from constructor timer interfere
    for attr in ("_inventory_task", "_indexing_task"):
        task = getattr(window, attr)
        if task is not None:
            task.cancel()
            thread = getattr(window, attr.replace("task", "thread"))
            if thread:
                thread.quit()
                thread.wait(3000)
            setattr(window, attr, None)
            setattr(window, attr.replace("task", "thread"), None)

    onboarding_called: list[bool] = []
    monkeypatch.setattr(
        window, "_show_onboarding", lambda: onboarding_called.append(True)
    )

    window._maybe_start_indexing()

    assert len(onboarding_called) == 0


def test_onboarding_accepted_saves_settings_and_starts_scan(
    qtbot, tmp_path, monkeypatch
):
    """_on_onboarding_accepted persists the path and kicks off the inventory scan."""
    collection_dir = tmp_path / "photos"
    collection_dir.mkdir()
    window = _make_window(tmp_path, qtbot)

    scan_called_with: list[str] = []
    monkeypatch.setattr(
        window, "_run_inventory_scan", lambda p: scan_called_with.append(p)
    )

    window._on_onboarding_accepted(str(collection_dir))

    assert window.settings.collection_path == str(collection_dir)
    assert scan_called_with == [str(collection_dir)]

    # Settings must be persisted to disk
    loaded = Settings.load(window.paths.config_file)
    assert loaded.collection_path == str(collection_dir)
