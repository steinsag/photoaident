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

    inventory_status = QtCore.Signal(str)
    inventory_progress = QtCore.Signal(int, int)
    inventory_finished = QtCore.Signal(int)
    indexing_progress = QtCore.Signal(int, int, int)
    indexing_finished = QtCore.Signal()

    def __init__(
        self,
        session_factory: sessionmaker,
        vector_store: VectorStore,
        paths: "AppPaths",
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._paths = paths

        self._inventory_task: InventoryTask | None = None
        self._inventory_thread: QtCore.QThread | None = None
        self._indexing_task: IndexingTask | None = None
        self._indexing_thread: QtCore.QThread | None = None

    @property
    def is_busy(self) -> bool:
        """True if any task (inventory or indexing) is currently running."""
        return self._inventory_task is not None or self._indexing_task is not None

    def start_pipeline(self, collection_path: str) -> None:
        """Run a silent inventory scan followed by indexing.

        No-op when already busy.
        """
        if self.is_busy:
            return
        self._start_inventory(collection_path)

    def start_indexing_only(self) -> None:
        """Skip inventory and start the indexing task directly.

        No-op when indexing is already running.
        """
        if self._indexing_task is not None:
            return
        self._start_indexing()

    def start_inventory(self, collection_path: str) -> None:
        """Run an inventory scan with progress reporting.

        No-op when already busy.
        """
        if self.is_busy:
            return
        self._inventory_task = InventoryTask(collection_path, self._session_factory)
        self._inventory_thread = QtCore.QThread()
        self._inventory_task.moveToThread(self._inventory_thread)

        self._inventory_task.status.connect(self.inventory_status)
        self._inventory_task.progress.connect(self.inventory_progress)
        self._inventory_task.finished.connect(
            self._on_inventory_finished_with_reporting
        )

        self._inventory_thread.started.connect(self._inventory_task.run)
        self._inventory_thread.finished.connect(self._inventory_thread.deleteLater)
        self._inventory_thread.start()

    def _on_inventory_finished_with_reporting(self, count: int) -> None:
        self._teardown_inventory()
        self.inventory_finished.emit(count)

    def cancel_inventory(self) -> None:
        """Stop the running inventory task immediately."""
        if self._inventory_task:
            self._inventory_task.cancel()

    def shutdown(self, faiss_path: Path) -> None:
        """Cancel all running tasks and persist the FAISS index.

        Call from ``closeEvent``; swallows exceptions to avoid blocking shutdown.
        """
        try:
            if self._inventory_task is not None:
                self._inventory_task.cancel()
            if self._inventory_thread is not None:
                self._inventory_thread.quit()

            if self._indexing_task is not None:
                self._indexing_task.cancel()
            if self._indexing_thread is not None:
                self._indexing_thread.quit()

            # Wait briefly for threads to stop, but don't block indefinitely
            if self._inventory_thread is not None:
                self._inventory_thread.wait(500)
            if self._indexing_thread is not None:
                self._indexing_thread.wait(1000)
        except Exception:
            logger.debug("Error during task cancellation", exc_info=True)

        if self._indexing_task is not None:
            try:
                self._vector_store.save(faiss_path)
            except Exception:
                logger.warning("Failed to save FAISS index on shutdown", exc_info=True)

    # ------------------------------------------------------------------
    # Private — inventory

    def _start_inventory(self, collection_path: str) -> None:
        self._inventory_task = InventoryTask(collection_path, self._session_factory)
        self._inventory_thread = QtCore.QThread()
        self._inventory_task.moveToThread(self._inventory_thread)
        self._inventory_task.finished.connect(self._on_inventory_finished)
        self._inventory_thread.started.connect(self._inventory_task.run)
        self._inventory_thread.finished.connect(self._inventory_thread.deleteLater)
        self._inventory_thread.start()

    def _on_inventory_finished(self, _: int) -> None:
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
        self._indexing_task = IndexingTask(
            self._session_factory, self._vector_store, self._paths
        )
        self._indexing_thread = QtCore.QThread()
        self._indexing_task.moveToThread(self._indexing_thread)

        self._indexing_task.progress.connect(self.indexing_progress)
        self._indexing_task.finished.connect(self._on_indexing_finished)
        self._indexing_thread.started.connect(self._indexing_task.run)
        self._indexing_thread.finished.connect(self._indexing_thread.deleteLater)
        self._indexing_thread.start()

    def _on_indexing_finished(self) -> None:
        self._teardown_indexing()
        self.indexing_finished.emit()

    def _teardown_indexing(self) -> None:
        if self._indexing_thread:
            self._indexing_thread.quit()
            self._indexing_thread.wait()
            self._indexing_thread = None
        self._indexing_task = None
