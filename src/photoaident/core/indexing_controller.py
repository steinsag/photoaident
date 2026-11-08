import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore
from sqlalchemy.orm import sessionmaker

from photoaident.core.indexing import IndexingTask
from photoaident.core.inventory import InventoryTask
from photoaident.db.vector_store import VectorStore

if TYPE_CHECKING:
    from photoaident.paths import AppPaths

logger = logging.getLogger(__name__)


class IndexingController(QtCore.QObject):
    """Owns the full inventory → indexing pipeline.

    Manages QThread lifecycle for both ``InventoryTask`` and ``IndexingTask``.
    ``MainWindow`` wires UI slots to the signals below; it never touches the
    task or thread objects directly.
    """

    inventory_progress = QtCore.Signal(int, int, str)
    inventory_finished = QtCore.Signal(int)
    indexing_progress = QtCore.Signal(int, int, int, str)
    indexing_finished = QtCore.Signal()

    def __init__(
        self,
        session_factory: sessionmaker,
        vector_store: VectorStore,
        paths: "AppPaths",
        filepath_date_pattern: str = "",
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._paths = paths
        self._filepath_date_pattern = filepath_date_pattern

        self._shutting_down = False
        self._shutdown_complete = False
        self._shutdown_in_progress = False
        self._inventory_task: InventoryTask | None = None
        self._inventory_thread: QtCore.QThread | None = None
        self._indexing_task: IndexingTask | None = None
        self._indexing_thread: QtCore.QThread | None = None

    @property
    def is_busy(self) -> bool:
        """True if any task (inventory or indexing) is currently running."""
        return self._inventory_task is not None or self._indexing_task is not None

    @property
    def filepath_date_pattern(self) -> str:
        """Current filepath date pattern used by new indexing tasks."""
        return self._filepath_date_pattern

    @filepath_date_pattern.setter
    def filepath_date_pattern(self, pattern: str) -> None:
        """Update the pattern used by subsequent indexing tasks."""
        self._filepath_date_pattern = pattern

    def start_pipeline(self, collection_path: str) -> None:
        """Run a silent inventory scan followed by indexing.

        No-op when already busy.
        """
        if self._shutting_down or self.is_busy:
            return
        self._start_inventory(collection_path)

    def start_indexing_only(self) -> None:
        """Skip inventory and start the indexing task directly.

        No-op when indexing is already running.
        """
        if self._shutting_down or self._indexing_task is not None:
            return
        self._start_indexing()

    def start_inventory(self, collection_path: str) -> None:
        """Run an inventory scan with progress reporting.

        No-op when already busy.
        """
        if self._shutting_down or self.is_busy:
            return
        self._inventory_task = InventoryTask(collection_path, self._session_factory)
        self._inventory_thread = QtCore.QThread()
        self._inventory_task.moveToThread(self._inventory_thread)

        self._inventory_task.progress.connect(self.inventory_progress)
        self._inventory_task.finished.connect(
            self._on_inventory_finished_with_reporting
        )

        self._inventory_thread.started.connect(self._inventory_task.run)
        self._inventory_thread.finished.connect(self._inventory_thread.deleteLater)
        self._inventory_thread.start()

    def _on_inventory_finished_with_reporting(self, count: int) -> None:
        if self._shutting_down:
            return
        self._teardown_inventory()
        self.inventory_finished.emit(count)

    def cancel_inventory(self) -> None:
        """Stop the running inventory task immediately."""
        if self._inventory_task:
            self._inventory_task.cancel()

    def shutdown(self, faiss_path: Path) -> None:
        """Cancel writers and join them before returning, keeping Qt responsive."""
        if self._shutdown_complete or self._shutdown_in_progress:
            return
        self._shutting_down = True
        self._shutdown_in_progress = True
        try:
            had_indexing = self._indexing_task is not None
            threads = [
                thread
                for thread in (self._inventory_thread, self._indexing_thread)
                if thread is not None
            ]
            # Keep thread wrappers alive while processing queued events during joins.
            for thread in threads:
                thread.finished.disconnect(thread.deleteLater)
            for task in (self._inventory_task, self._indexing_task):
                if task is not None:
                    try:
                        task.cancel()
                    except Exception:
                        logger.warning(
                            "Task cancellation failed; waiting for completion",
                            exc_info=True,
                        )
            for thread in threads:
                thread.quit()
            for thread in threads:
                while not thread.wait(50):
                    QtCore.QCoreApplication.processEvents(
                        QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents, 50
                    )
                thread.deleteLater()
            self._inventory_thread = None
            self._indexing_thread = None
            self._inventory_task = None
            self._indexing_task = None
            if had_indexing:
                try:
                    self._vector_store.save(faiss_path)
                except Exception:
                    logger.warning(
                        "Failed to save FAISS index on shutdown", exc_info=True
                    )

            self._shutdown_complete = True
        finally:
            self._shutdown_in_progress = False

    # ------------------------------------------------------------------
    # Private — inventory

    def _start_inventory(self, collection_path: str) -> None:
        if self._shutting_down:
            return
        self._inventory_task = InventoryTask(collection_path, self._session_factory)
        self._inventory_thread = QtCore.QThread()
        self._inventory_task.moveToThread(self._inventory_thread)
        self._inventory_task.finished.connect(self._on_inventory_finished)
        self._inventory_thread.started.connect(self._inventory_task.run)
        self._inventory_thread.finished.connect(self._inventory_thread.deleteLater)
        self._inventory_thread.start()

    def _on_inventory_finished(self, _: int) -> None:
        if self._shutting_down:
            return
        self._teardown_inventory()
        self._start_indexing()

    def _teardown_inventory(self) -> None:
        if self._inventory_thread:
            self._inventory_thread.quit()
            self._inventory_thread.wait()
            self._inventory_thread = None
        self._inventory_task = None

    # ------------------------------------------------------------------
    # Private — indexing

    def _start_indexing(self) -> None:
        if self._shutting_down:
            return
        self._indexing_task = IndexingTask(
            self._session_factory,
            self._vector_store,
            self._paths,
            filepath_date_pattern=self._filepath_date_pattern,
        )
        self._indexing_thread = QtCore.QThread()
        self._indexing_task.moveToThread(self._indexing_thread)

        self._indexing_task.progress.connect(self.indexing_progress)
        self._indexing_task.finished.connect(self._on_indexing_finished)
        self._indexing_thread.started.connect(self._indexing_task.run)
        self._indexing_thread.finished.connect(self._indexing_thread.deleteLater)
        self._indexing_thread.start()

    def _on_indexing_finished(self) -> None:
        if self._shutting_down:
            return
        self._teardown_indexing()
        self.indexing_finished.emit()

    def _teardown_indexing(self) -> None:
        if self._indexing_thread:
            self._indexing_thread.quit()
            self._indexing_thread.wait()
            self._indexing_thread = None
        self._indexing_task = None
