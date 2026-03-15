"""Tests for IndexingController.

Strategy: inject mock tasks/threads directly into private attrs; call private
methods to simulate task-completion signals; never start real threads.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from photoaident.core.indexing_controller import IndexingController
from photoaident.db.vector_store import VectorStore


def _make_controller(
    tmp_path: Path,
) -> tuple[IndexingController, MagicMock]:
    """Create an IndexingController with mock dependencies.

    Returns ``(controller, vector_store_mock)`` so callers can assert on the mock.
    """
    session_factory = MagicMock()
    vector_store: MagicMock = MagicMock(spec=VectorStore)
    paths = MagicMock()
    paths.faiss_path = tmp_path / "faiss.index"
    controller = IndexingController(session_factory, vector_store, paths)
    return controller, vector_store


# ---------------------------------------------------------------------------
# is_busy property
# ---------------------------------------------------------------------------


def test_not_busy_initially(tmp_path):
    """is_busy is False when no tasks have been started."""
    controller, _ = _make_controller(tmp_path)
    assert not controller.is_busy


def test_busy_when_inventory_task_set(tmp_path):
    """is_busy is True when an inventory task is active."""
    controller, _ = _make_controller(tmp_path)
    controller._inventory_task = MagicMock()
    assert controller.is_busy


def test_busy_when_indexing_task_set(tmp_path):
    """is_busy is True when an indexing task is active."""
    controller, _ = _make_controller(tmp_path)
    controller._indexing_task = MagicMock()
    assert controller.is_busy


# ---------------------------------------------------------------------------
# start_pipeline
# ---------------------------------------------------------------------------


def test_start_pipeline_creates_inventory_task(tmp_path):
    """start_pipeline starts an InventoryTask thread."""
    controller, _ = _make_controller(tmp_path)

    with (
        patch("photoaident.core.indexing_controller.InventoryTask") as MockTask,
        patch("photoaident.core.indexing_controller.QtCore.QThread") as MockThread,
    ):
        mock_task = MagicMock()
        MockTask.return_value = mock_task
        mock_thread = MagicMock()
        MockThread.return_value = mock_thread

        controller.start_pipeline("/photos")

    assert controller._inventory_task is mock_task
    mock_thread.start.assert_called_once()


def test_start_pipeline_idempotent_when_busy(tmp_path):
    """start_pipeline is a no-op when a task is already running."""
    controller, _ = _make_controller(tmp_path)
    existing_task = MagicMock()
    controller._inventory_task = existing_task

    with patch("photoaident.core.indexing_controller.InventoryTask") as MockTask:
        controller.start_pipeline("/photos")
        MockTask.assert_not_called()

    assert controller._inventory_task is existing_task


# ---------------------------------------------------------------------------
# start_indexing_only
# ---------------------------------------------------------------------------


def test_start_indexing_only_creates_indexing_task(tmp_path):
    """start_indexing_only creates and starts an IndexingTask."""
    controller, _ = _make_controller(tmp_path)

    with (
        patch("photoaident.core.indexing_controller.IndexingTask") as MockTask,
        patch("photoaident.core.indexing_controller.QtCore.QThread") as MockThread,
    ):
        mock_task = MagicMock()
        MockTask.return_value = mock_task
        mock_thread = MagicMock()
        MockThread.return_value = mock_thread

        controller.start_indexing_only()

    assert controller._indexing_task is mock_task
    mock_thread.start.assert_called_once()


def test_start_indexing_only_idempotent_when_indexing_running(tmp_path):
    """start_indexing_only is a no-op when an indexing task is already running."""
    controller, _ = _make_controller(tmp_path)
    existing_task = MagicMock()
    controller._indexing_task = existing_task

    with patch("photoaident.core.indexing_controller.IndexingTask") as MockTask:
        controller.start_indexing_only()
        MockTask.assert_not_called()

    assert controller._indexing_task is existing_task


# ---------------------------------------------------------------------------
# _on_inventory_finished → clears inventory and starts indexing
# ---------------------------------------------------------------------------


def test_on_inventory_finished_clears_task_and_starts_indexing(tmp_path):
    """_on_inventory_finished tears down the inventory and starts indexing."""
    controller, _ = _make_controller(tmp_path)

    mock_thread = MagicMock()
    controller._inventory_task = MagicMock()
    controller._inventory_thread = mock_thread

    with patch.object(controller, "_start_indexing") as mock_start_indexing:
        controller._on_inventory_finished(5)

    mock_thread.quit.assert_called_once()
    mock_thread.wait.assert_called_once()
    assert controller._inventory_task is None
    assert controller._inventory_thread is None
    mock_start_indexing.assert_called_once()


# ---------------------------------------------------------------------------
# _on_indexing_finished → emits signal
# ---------------------------------------------------------------------------


def test_on_indexing_finished_emits_signal_and_clears_task(tmp_path):
    """_on_indexing_finished emits indexing_finished and clears the task."""
    controller, _ = _make_controller(tmp_path)

    mock_thread = MagicMock()
    controller._indexing_task = MagicMock()
    controller._indexing_thread = mock_thread

    emitted: list[bool] = []
    controller.indexing_finished.connect(lambda: emitted.append(True))

    controller._on_indexing_finished()

    assert emitted == [True]
    assert controller._indexing_task is None
    assert controller._indexing_thread is None


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


def test_shutdown_cancels_inventory_task(tmp_path):
    """shutdown() cancels a running inventory task."""
    controller, _ = _make_controller(tmp_path)

    mock_task = MagicMock()
    mock_thread = MagicMock()
    controller._inventory_task = mock_task
    controller._inventory_thread = mock_thread

    controller.shutdown(tmp_path / "faiss.index")

    mock_task.cancel.assert_called_once()
    mock_thread.quit.assert_called_once()


def test_shutdown_cancels_indexing_and_saves_faiss(tmp_path):
    """shutdown() cancels the indexing task and persists the FAISS index."""
    controller, vector_store = _make_controller(tmp_path)

    mock_task = MagicMock()
    mock_thread = MagicMock()
    controller._indexing_task = mock_task
    controller._indexing_thread = mock_thread

    faiss_path = tmp_path / "faiss.index"
    controller.shutdown(faiss_path)

    mock_task.cancel.assert_called_once()
    mock_thread.quit.assert_called_once()
    vector_store.save.assert_called_once_with(faiss_path)


def test_shutdown_handles_cancel_exception_gracefully(tmp_path):
    """shutdown() swallows exceptions from cancel() and vector_store.save()."""
    controller, vector_store = _make_controller(tmp_path)

    mock_task = MagicMock()
    mock_task.cancel.side_effect = RuntimeError("already done")
    mock_thread = MagicMock()
    controller._indexing_task = mock_task
    controller._indexing_thread = mock_thread
    vector_store.save.side_effect = OSError("disk full")

    # Must not raise
    controller.shutdown(tmp_path / "faiss.index")


def test_shutdown_no_tasks_is_noop(tmp_path):
    """shutdown() with no running tasks does not save the FAISS index."""
    controller, vector_store = _make_controller(tmp_path)
    # Must not raise; vector_store.save should NOT be called (no indexing task)
    controller.shutdown(tmp_path / "faiss.index")
    vector_store.save.assert_not_called()


def test_shutdown_logs_warning_when_faiss_save_fails(tmp_path):
    """shutdown() logs a warning when vector_store.save raises, does not propagate."""
    controller, vector_store = _make_controller(tmp_path)

    mock_task = MagicMock()
    mock_thread = MagicMock()
    mock_thread.wait.return_value = True  # thread stopped in time
    controller._indexing_task = mock_task
    controller._indexing_thread = mock_thread
    vector_store.save.side_effect = OSError("disk full")

    # Must not raise despite save() failing
    controller.shutdown(tmp_path / "faiss.index")
    vector_store.save.assert_called_once()


def test_start_inventory_noop_when_busy(tmp_path):
    """start_inventory() is a no-op when the controller is already busy."""
    controller, _ = _make_controller(tmp_path)

    # Simulate a running inventory task so is_busy returns True
    controller._inventory_task = MagicMock()

    with patch.object(controller, "_start_inventory") as mock_start:
        controller.start_inventory("some/path")

    mock_start.assert_not_called()
