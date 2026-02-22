import fcntl
import os
from pathlib import Path
from typing import TextIO


class InstanceLock:
    """Manages a file-based lock to prevent multiple instances of the app."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._lock_file: TextIO | None = None

    def acquire(self) -> bool:
        """Try to acquire the lock.

        Returns:
            True if successful, False if already locked.
        """
        try:
            # Ensure parent directory exists
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)

            # Open the lock file
            self._lock_file = open(self.lock_path, "w")

            # Try to get an exclusive lock (non-blocking)
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write current PID to the lock file for debugging
            self._lock_file.write(str(os.getpid()))
            self._lock_file.flush()

            return True
        except (IOError, OSError):
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._lock_file:
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            except (IOError, OSError):
                pass
            finally:
                self._lock_file.close()
                self._lock_file = None
                # Optionally delete the file, but flock is usually enough
                try:
                    self.lock_path.unlink(missing_ok=True)
                except (IOError, OSError):
                    pass
