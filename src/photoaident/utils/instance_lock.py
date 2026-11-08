import logging
from pathlib import Path

from PySide6.QtCore import QLockFile

logger = logging.getLogger(__name__)


class InstanceLock:
    """Hold a non-recursive application lock without requiring a Qt event loop."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._resolved_path = lock_path.absolute()
        self._lock = QLockFile(str(self._resolved_path))
        # A live application may index photos for hours.
        self._lock.setStaleLockTime(0)

    def acquire(self) -> bool:
        """Acquire immediately, returning False on contention or filesystem errors."""
        if self._lock.isLocked():
            return False
        try:
            self._resolved_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning(
                "Cannot create lock directory for %s", self.lock_path, exc_info=True
            )
            return False
        # Legacy PID-only files lack Qt owner metadata. Do not ask Qt to
        # interpret a truncated PID as evidence that such an owner is dead.
        try:
            with self._resolved_path.open("rb") as existing:
                metadata = [existing.readline(4096) for _ in range(3)]
            if any(not line.endswith(b"\n") for line in metadata):
                logger.warning(
                    "Unrecognized instance lock metadata: %s", self.lock_path
                )
                return False
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning(
                "Cannot read instance lock %s", self.lock_path, exc_info=True
            )
            return False
        if self._lock.tryLock(0):
            return True
        error = self._lock.error()
        if error != QLockFile.LockError.LockFailedError:
            logger.warning("Cannot acquire instance lock %s: %s", self.lock_path, error)
        return False

    def release(self) -> None:
        """Release only this instance's ownership; repeated calls are harmless."""
        self._lock.unlock()
