import builtins
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import onnxruntime as ort
import pytest
from PySide6 import QtGui, QtWidgets

import photoaident.app as app_module
from photoaident.app import MainWindow, get_resource_path, load_translations
from photoaident.db.database import Face, Image
from photoaident.db.migrate import apply_migrations
from photoaident.db.vector_store import VectorStore
from photoaident.paths import AppPaths
from photoaident.settings import Settings


def test_navigation_shortcuts(qtbot, tmp_app_paths):
    """Test that Alt+1 to Alt+4 shortcuts switch to the correct page."""
    window = _make_window(tmp_app_paths, qtbot)

    # Verify shortcuts are set correctly
    assert window.btn_search.shortcut().toString() == "Alt+1"
    assert window.btn_browse.shortcut().toString() == "Alt+2"
    assert window.btn_persons.shortcut().toString() == "Alt+3"
    assert window.btn_label.shortcut().toString() == "Alt+4"

    # Initial page should be 0 (Search)
    assert window.stacked.currentIndex() == 0

    # Switch to Browse (Alt+2)
    window.btn_browse.animateClick()
    qtbot.wait(100)
    assert window.stacked.currentIndex() == 1

    # Switch to Persons (Alt+3)
    window.btn_persons.animateClick()
    qtbot.wait(100)
    assert window.stacked.currentIndex() == 2

    # Switch to Labelling (Alt+4)
    window.btn_label.animateClick()
    qtbot.wait(100)
    assert window.stacked.currentIndex() == 3

    # Switch back to Search (Alt+1)
    window.btn_search.animateClick()
    qtbot.wait(100)
    assert window.stacked.currentIndex() == 0


def test_app_setup(qtbot, tmp_app_paths):
    """
    Test that the application window can be instantiated.
    """
    window = _make_window(tmp_app_paths, qtbot)

    assert window.windowTitle() == "PhotoAIdent"
    assert window.library_page is not None
    assert window.labelling_page is not None


def test_startup_triggers_inventory_scan_when_collection_path_set(qtbot, tmp_app_paths):
    """_maybe_start_indexing starts an InventoryTask when a collection path is set."""
    collection_dir = tmp_app_paths.thumbs_dir / "photos"
    collection_dir.mkdir()

    window = _make_window(tmp_app_paths, qtbot, collection_path=str(collection_dir))
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


def test_startup_skips_scan_when_no_collection_path(qtbot, tmp_app_paths):
    """_maybe_start_indexing does nothing when collection_path is empty."""
    window = _make_window(tmp_app_paths, qtbot, collection_path="")
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


def test_startup_scan_idempotent_when_already_running(qtbot, tmp_app_paths):
    """Calling _maybe_start_indexing twice does not start a second inventory."""
    collection_dir = _make_collection_dir(tmp_app_paths)

    window = _make_window(tmp_app_paths, qtbot, collection_path=str(collection_dir))
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


def test_counts_label_shows_zeros_on_empty_db(qtbot, tmp_app_paths):
    """counts_label shows 0 images and 0 faces when the database is empty."""
    window = _make_window(tmp_app_paths, qtbot)
    text = window.counts_label.text()
    assert "0" in text


def test_counts_label_reflects_db_contents(qtbot, tmp_app_paths):
    """counts_label shows the correct totals loaded from the database."""
    window = _make_window(tmp_app_paths, qtbot)

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


def test_counts_label_updated_after_indexing_finished(qtbot, tmp_app_paths):
    """_on_indexing_finished refreshes the counts label."""
    window = _make_window(tmp_app_paths, qtbot)

    with window.session_factory() as session:
        img = Image(file_path="/test/photo2.jpg", file_hash="def456", file_size=500)
        session.add(img)
        session.commit()

    window._on_indexing_finished()

    text = window.counts_label.text()
    assert "1" in text  # 1 image


def test_onboarding_triggered_when_no_path(qtbot, tmp_app_paths, monkeypatch):
    """_maybe_start_indexing triggers onboarding when no collection path is set."""
    window = _make_window(tmp_app_paths, qtbot)
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


def test_onboarding_not_triggered_when_path_set(qtbot, tmp_app_paths, monkeypatch):
    """_maybe_start_indexing does not trigger onboarding when a path is already set."""
    collection_dir = _make_collection_dir(tmp_app_paths)

    window = _make_window(tmp_app_paths, qtbot, collection_path=str(collection_dir))
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
    qtbot, tmp_app_paths, monkeypatch
):
    """_on_onboarding_accepted persists the path and kicks off the inventory scan."""
    collection_dir = _make_collection_dir(tmp_app_paths)

    window = _make_window(tmp_app_paths, qtbot)

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


# ---------------------------------------------------------------------------
# get_resource_path — PyInstaller bundle branch (line 34)
# ---------------------------------------------------------------------------


def test_get_resource_path_pyinstaller_bundle(monkeypatch):
    """Uses _MEIPASS as base when running from a PyInstaller bundle."""
    monkeypatch.setattr(sys, "_MEIPASS", "/bundle_root", raising=False)
    result = get_resource_path("assets/icon.png")
    assert result == os.path.join("/bundle_root", "assets/icon.png")


# ---------------------------------------------------------------------------
# load_translations (lines 46-65)
# ---------------------------------------------------------------------------


def test_load_translations_no_match_is_silent(monkeypatch):
    """load_translations is a no-op when no .qm file matches the locale."""
    monkeypatch.setattr(app_module, "get_resource_path", lambda p: "/no/such/file.qm")
    qt_app = QtWidgets.QApplication.instance()
    assert isinstance(qt_app, QtWidgets.QApplication)
    load_translations(qt_app)  # must not raise


def test_load_translations_installs_translator(monkeypatch):
    """load_translations installs a QTranslator when a valid .qm file is found."""
    import pathlib

    project_root = pathlib.Path(__file__).parent.parent
    qm_path = project_root / "assets" / "translations" / "photoaident_de.qm"
    if not qm_path.exists():
        pytest.skip("photoaident_de.qm not built yet")

    # Always return the German .qm so the locale-based loop hits the file
    monkeypatch.setattr(app_module, "get_resource_path", lambda p: str(qm_path))
    qt_app = QtWidgets.QApplication.instance()
    assert isinstance(qt_app, QtWidgets.QApplication)
    qt_app.__dict__.pop("_translator", None)
    load_translations(qt_app)
    try:
        assert hasattr(qt_app, "_translator")
    finally:
        # Remove the translator so it does not affect other tests in the session
        translator = qt_app.__dict__.pop("_translator", None)
        if translator is not None:
            qt_app.removeTranslator(translator)


# ---------------------------------------------------------------------------
# vector_store.load at startup (line 91)
# ---------------------------------------------------------------------------


def test_vector_store_loaded_when_faiss_file_exists(tmp_app_paths, qtbot):
    """MainWindow loads an existing FAISS index if one is present at startup."""
    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")

    # Pre-create a FAISS index on disk
    vs = VectorStore()
    vs.save(tmp_app_paths.faiss_path)
    assert tmp_app_paths.faiss_path.exists()

    window = MainWindow(tmp_app_paths, check_gpu=False, enable_onboarding=False)
    qtbot.addWidget(window)

    assert window.vector_store is not None


# ---------------------------------------------------------------------------
# GPU check thread (line 163)
# ---------------------------------------------------------------------------


def test_gpu_check_thread_started_when_enabled(
    tmp_app_paths: AppPaths, qtbot, monkeypatch
):
    """check_gpu=True starts a background thread that calls _check_gpu."""
    event = threading.Event()

    monkeypatch.setattr(MainWindow, "_check_gpu", lambda self: event.set())

    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")

    window = MainWindow(tmp_app_paths, check_gpu=True, enable_onboarding=False)
    qtbot.addWidget(window)

    assert event.wait(timeout=2.0), "_check_gpu was not called within 2 s"


# ---------------------------------------------------------------------------
# _show_onboarding dialog (lines 175-225)
# ---------------------------------------------------------------------------


def test_show_onboarding_cancelled_does_nothing(tmp_app_paths, qtbot, monkeypatch):
    """_show_onboarding returns early without saving anything if cancelled."""
    window = _make_window(tmp_app_paths, qtbot)

    monkeypatch.setattr(
        QtWidgets.QDialog,
        "exec",
        lambda _self: QtWidgets.QDialog.DialogCode.Rejected,
    )
    called: list[str] = []
    monkeypatch.setattr(window, "_on_onboarding_accepted", lambda p: called.append(p))

    window._show_onboarding()

    assert called == []


def test_show_onboarding_accepted_calls_handler(tmp_app_paths, qtbot, monkeypatch):
    """_show_onboarding calls _on_onboarding_accepted with the chosen folder."""
    collection_dir = _make_collection_dir(tmp_app_paths)

    window = _make_window(tmp_app_paths, qtbot)

    def fake_exec(dialog_self: QtWidgets.QDialog) -> int:
        line_edit = dialog_self.findChild(QtWidgets.QLineEdit)
        if line_edit:
            line_edit.setText(str(collection_dir))
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_exec)
    called: list[str] = []
    monkeypatch.setattr(window, "_on_onboarding_accepted", lambda p: called.append(p))

    window._show_onboarding()

    assert called == [str(collection_dir)]


# ---------------------------------------------------------------------------
# _start_indexing_task guard clause (line 265)
# ---------------------------------------------------------------------------


def test_start_indexing_task_idempotent(tmp_app_paths, qtbot):
    """_start_indexing_task is a no-op when a task is already running."""
    window = _make_window(tmp_app_paths, qtbot)

    mock_task = MagicMock()
    window._indexing_task = mock_task

    window._start_indexing_task()

    # Task object must not have been replaced
    assert window._indexing_task is mock_task


# ---------------------------------------------------------------------------
# _update_indexing_status (lines 280-287)
# ---------------------------------------------------------------------------


def test_update_indexing_status_formats_label_and_reloads(
    tmp_app_paths, qtbot, monkeypatch
):
    """_update_indexing_status updates the label and reloads every 50 images."""
    window = _make_window(tmp_app_paths, qtbot)
    reloads: list[bool] = []
    monkeypatch.setattr(
        window.library_page, "load_images", lambda: reloads.append(True)
    )

    window._update_indexing_status(50, 100, 7)
    assert "50" in window.indexing_label.text()
    assert "100" in window.indexing_label.text()
    assert "7" in window.indexing_label.text()
    assert len(reloads) == 1  # 50 % 50 == 0 → reload

    window._update_indexing_status(51, 100, 7)
    assert len(reloads) == 1  # 51 % 50 != 0 and 51 != 100 → no reload

    window._update_indexing_status(100, 100, 7)
    assert len(reloads) == 2  # indexed == total → reload


# ---------------------------------------------------------------------------
# _switch_page labelling branch (line 317)
# ---------------------------------------------------------------------------


def test_switch_page_to_label_refreshes_labelling_page(
    tmp_app_paths, qtbot, monkeypatch
):
    """_switch_page(3) calls labelling_page.refresh()."""
    window = _make_window(tmp_app_paths, qtbot)
    refreshed: list[bool] = []
    monkeypatch.setattr(
        window.labelling_page, "refresh", lambda: refreshed.append(True)
    )

    window._switch_page(3)

    assert window.stacked.currentIndex() == 3
    assert len(refreshed) == 1


# ---------------------------------------------------------------------------
# go_to_labelling (lines 323-329)
# ---------------------------------------------------------------------------


def test_go_to_labelling_navigates_and_passes_priority(
    tmp_app_paths, qtbot, monkeypatch
):
    """go_to_labelling switches to page 3 and forwards the image id."""
    window = _make_window(tmp_app_paths, qtbot)
    refreshed_with: list[int | None] = []
    monkeypatch.setattr(
        window.labelling_page,
        "refresh",
        lambda priority_image_id=None: refreshed_with.append(priority_image_id),
    )

    window.go_to_labelling(42)

    assert window.stacked.currentIndex() == 3
    assert window.btn_label.isChecked()
    assert refreshed_with == [42]


# ---------------------------------------------------------------------------
# _show_about (lines 358-359)
# ---------------------------------------------------------------------------


def test_show_about_opens_dialog(tmp_app_paths, qtbot, monkeypatch):
    """_show_about creates and executes an AboutDialog without raising."""
    monkeypatch.setattr(app_module, "AboutDialog", lambda _parent: MagicMock())
    window = _make_window(tmp_app_paths, qtbot)
    window._show_about()  # must not raise


# ---------------------------------------------------------------------------
# _show_preferences — path unchanged branch (line 409)
# ---------------------------------------------------------------------------


def test_show_preferences_saves_settings_when_path_unchanged(
    tmp_app_paths, qtbot, monkeypatch
):
    """_show_preferences saves settings even when the collection path is unchanged."""
    collection_dir = _make_collection_dir(tmp_app_paths)
    window = _make_window(tmp_app_paths, qtbot, collection_path=str(collection_dir))

    class FakePreferencesDialog:
        def __init__(self, *_args, **_kwargs):
            # Minimal mock initialization for PreferencesDialog used in tests
            pass

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Accepted

        def get_collection_path(self) -> str:
            return str(collection_dir)  # same as old path

    monkeypatch.setattr(app_module, "PreferencesDialog", FakePreferencesDialog)

    window._show_preferences()

    loaded = Settings.load(window.paths.config_file)
    assert loaded.collection_path == str(collection_dir)


# ---------------------------------------------------------------------------
# _check_gpu (lines 443-464)
# ---------------------------------------------------------------------------


def test_check_gpu_cuda_available(tmp_app_paths, qtbot, monkeypatch):
    """_check_gpu emits a GPU-ready message when CUDAExecutionProvider is present."""
    window = _make_window(tmp_app_paths, qtbot)

    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    # Ensure insightface is importable (use cached module or a stub)
    if "insightface" not in sys.modules:
        monkeypatch.setitem(sys.modules, "insightface", MagicMock())

    captured: list[str] = []
    window._gpu_status_signal.status_ready.connect(lambda msg: captured.append(msg))
    window._check_gpu()

    assert captured
    assert "GPU" in captured[0] or "✅" in captured[0]


def test_check_gpu_cpu_only(tmp_app_paths, qtbot, monkeypatch):
    """_check_gpu emits a CPU-only warning when CUDA is not available."""
    window = _make_window(tmp_app_paths, qtbot)

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    if "insightface" not in sys.modules:
        monkeypatch.setitem(sys.modules, "insightface", MagicMock())

    captured: list[str] = []
    window._gpu_status_signal.status_ready.connect(lambda msg: captured.append(msg))
    window._check_gpu()

    assert captured
    assert "CPU" in captured[0] or "⚠️" in captured[0]


def test_check_gpu_import_error(tmp_app_paths, qtbot, monkeypatch):
    """_check_gpu emits an error message when insightface cannot be imported."""
    window = _make_window(tmp_app_paths, qtbot)

    real_import = builtins.__import__

    def mock_import(name: str, *args, **kwargs):
        if name == "insightface":
            raise ImportError("mocked: insightface not available")
        return real_import(name, *args, **kwargs)

    # Remove cached module so our __import__ hook is reached
    monkeypatch.delitem(sys.modules, "insightface", raising=False)
    monkeypatch.setattr(builtins, "__import__", mock_import)

    captured: list[str] = []
    window._gpu_status_signal.status_ready.connect(lambda msg: captured.append(msg))
    window._check_gpu()

    assert captured
    assert "❌" in captured[0]


# ---------------------------------------------------------------------------
# _on_gpu_status_ready (line 467)
# ---------------------------------------------------------------------------


def test_on_gpu_status_ready_shows_message(tmp_app_paths, qtbot):
    """_on_gpu_status_ready forwards the message to the status bar."""
    window = _make_window(tmp_app_paths, qtbot)
    window._on_gpu_status_ready("Test GPU message")
    # showMessage is timed; just verify no exception is raised


# ---------------------------------------------------------------------------
# closeEvent cleanup (lines 472-477, 484-485, 490-491)
# ---------------------------------------------------------------------------


def test_close_event_cancels_inventory_task(tmp_app_paths, qtbot):
    """closeEvent cancels a running inventory task and accepts the event."""
    window = _make_window(tmp_app_paths, qtbot)

    mock_task = MagicMock()
    mock_thread = MagicMock()
    window._inventory_task = mock_task
    window._inventory_thread = mock_thread

    event = MagicMock(spec=QtGui.QCloseEvent)
    window.closeEvent(event)

    mock_task.cancel.assert_called_once()
    mock_thread.quit.assert_called_once()
    event.accept.assert_called_once()


def test_close_event_cancels_indexing_task_and_saves_vector_store(tmp_app_paths, qtbot):
    """closeEvent cancels a running indexing task and persists the FAISS index."""
    window = _make_window(tmp_app_paths, qtbot)

    mock_task = MagicMock()
    mock_thread = MagicMock()
    window._indexing_task = mock_task
    window._indexing_thread = mock_thread

    event = MagicMock(spec=QtGui.QCloseEvent)
    window.closeEvent(event)

    mock_task.cancel.assert_called_once()
    mock_thread.quit.assert_called_once()
    assert window.paths.faiss_path.exists()
    event.accept.assert_called_once()


def test_close_event_handles_exceptions_gracefully(tmp_app_paths, qtbot, monkeypatch):
    """closeEvent swallows exceptions from task.cancel() and vector_store.save()."""
    window = _make_window(tmp_app_paths, qtbot)

    mock_task = MagicMock()
    mock_task.cancel.side_effect = RuntimeError("already done")
    mock_thread = MagicMock()
    window._indexing_task = mock_task
    window._indexing_thread = mock_thread
    monkeypatch.setattr(
        window.vector_store, "save", MagicMock(side_effect=OSError("disk full"))
    )

    event = MagicMock(spec=QtGui.QCloseEvent)
    window.closeEvent(event)  # must not raise

    event.accept.assert_called_once()


def _make_window(
    tmp_app_paths: AppPaths, qtbot, collection_path: str = ""
) -> MainWindow:
    """Helper: create a MainWindow with migrations applied. Onboarding is disabled."""
    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")
    window = MainWindow(tmp_app_paths, check_gpu=False, enable_onboarding=False)
    window.settings.collection_path = collection_path
    qtbot.addWidget(window)
    return window


def _make_collection_dir(tmp_app_paths: AppPaths) -> Path:
    collection_dir = tmp_app_paths.thumbs_dir / "photos"
    collection_dir.mkdir(exist_ok=True)
    return collection_dir
