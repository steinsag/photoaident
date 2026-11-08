"""Cross-platform instance ownership, including independent process races."""

import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QLockFile

from photoaident.utils.instance_lock import InstanceLock

_CHILD = """
import sys
from pathlib import Path
from PySide6.QtCore import QCoreApplication
from photoaident.utils.instance_lock import InstanceLock
app = QCoreApplication([]) if sys.argv[2] == "core" else None
assert (QCoreApplication.instance() is not None) == (app is not None)
lock = InstanceLock(Path(sys.argv[1]))
print("READY", flush=True)
for command in sys.stdin:
    command = command.strip()
    if command == "ACQUIRE":
        print("ACQUIRED" if lock.acquire() else "REJECTED", flush=True)
    elif command == "RELEASE":
        lock.release()
        print("RELEASED", flush=True)
    elif command == "EXIT":
        lock.release()
        break
"""


@contextmanager
def child(
    lock_path: Path, mode: str = "none"
) -> Iterator[tuple[subprocess.Popen[str], Callable[[str], None], Callable[[], str]]]:
    """Start a fresh interpreter and always reap it, even after failed assertions."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", _CHILD, str(lock_path), mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    messages: queue.Queue[str] = queue.Queue()
    assert process.stdout is not None
    output = process.stdout

    def read_messages() -> None:
        for line in output:
            messages.put(line.strip())

    reader = threading.Thread(target=read_messages, daemon=True)
    reader.start()

    def receive() -> str:
        try:
            return messages.get(timeout=10)
        except queue.Empty:
            pytest.fail(f"Child did not respond; exit code={process.poll()}")

    def send(command: str) -> None:
        assert process.stdin is not None
        process.stdin.write(command + "\n")
        process.stdin.flush()

    try:
        assert receive() == "READY"
        yield process, send, receive
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        reader.join(timeout=10)


def test_acquire_release_reacquire(tmp_path):
    """Only one wrapper owns the lock and ownership can be reused."""
    path = tmp_path / "nested" / "photoaident.lock"
    first, second = InstanceLock(path), InstanceLock(path)
    try:
        assert first.acquire()
        contents = path.read_bytes()
        assert not first.acquire()
        assert not second.acquire()
        second.release()
        assert path.read_bytes() == contents
        first.release()
        first.release()
        assert not path.exists()
        assert second.acquire()
        second.release()
        assert first.acquire()
    finally:
        first.release()
        second.release()


@pytest.mark.parametrize("content", ["", "not a lock", "123456"])
def test_ambiguous_files_are_not_removed(tmp_path, content):
    """Unknown owners are never evicted merely because a file is old."""
    path = tmp_path / "photoaident.lock"
    path.write_text(content)
    os.utime(path, (1, 1))
    lock = InstanceLock(path)
    assert not lock.acquire()
    lock.release()
    assert path.read_text() == content


def test_parent_creation_error(tmp_path, caplog):
    """Filesystem errors fail closed and are logged."""
    lock = InstanceLock(tmp_path / "photoaident.lock")
    with patch.object(Path, "mkdir", side_effect=OSError("denied")):
        assert not lock.acquire()
    assert "Cannot create lock directory" in caplog.text


@pytest.mark.parametrize(
    "error", [QLockFile.LockError.PermissionError, QLockFile.LockError.UnknownError]
)
def test_qt_errors_are_logged(tmp_path, caplog, error):
    """Operational Qt failures retain the boolean interface."""
    with patch("photoaident.utils.instance_lock.QLockFile") as backend:
        native = MagicMock()
        backend.return_value = native
        native.isLocked.return_value = False
        native.tryLock.return_value = False
        native.error.return_value = error
        lock = InstanceLock(tmp_path / "photoaident.lock")
        assert not lock.acquire()
        native.setStaleLockTime.assert_called_once_with(0)
        native.tryLock.assert_called_once_with(0)
    assert "Cannot acquire instance lock" in caplog.text


@pytest.mark.parametrize("mode", ["none", "core"])
def test_no_running_event_loop(tmp_path, mode):
    """Neither a QApplication nor exec() is needed."""
    with child(tmp_path / "photoaident.lock", mode) as (process, send, receive):
        send("ACQUIRE")
        assert receive() == "ACQUIRED"
        send("RELEASE")
        assert receive() == "RELEASED"
        send("EXIT")
        assert process.wait(timeout=10) == 0


@pytest.mark.parametrize("attempt", range(3))
def test_process_race_and_handover(tmp_path, attempt):
    """Simultaneous contenders cannot both win; release permits a live handover."""
    path = tmp_path / f"árvíz space {attempt}" / "photoaident.lock"
    with child(path) as first, child(path) as second:
        first[1]("ACQUIRE")
        second[1]("ACQUIRE")
        results = [first[2](), second[2]()]
        assert sorted(results) == ["ACQUIRED", "REJECTED"]
        owner, loser = (first, second) if results[0] == "ACQUIRED" else (second, first)
        os.utime(path, (1, 1))
        loser[1]("ACQUIRE")
        assert loser[2]() == "REJECTED"
        owner[1]("RELEASE")
        assert owner[2]() == "RELEASED"
        assert owner[0].poll() is None
        loser[1]("ACQUIRE")
        assert loser[2]() == "ACQUIRED"
        loser[1]("RELEASE")
        assert loser[2]() == "RELEASED"
        assert not path.exists()


def test_crash_recovery_race(tmp_path):
    """OS termination leaves metadata that only one fresh process can recover."""
    path = tmp_path / "photoaident.lock"
    with child(path) as (process, send, receive):
        send("ACQUIRE")
        assert receive() == "ACQUIRED"
        process.kill()
        process.wait(timeout=10)
        assert path.exists()
        with child(path) as first, child(path) as second:
            first[1]("ACQUIRE")
            second[1]("ACQUIRE")
            assert sorted([first[2](), second[2]()]) == ["ACQUIRED", "REJECTED"]


def test_relative_path_is_stable_after_chdir(tmp_path, monkeypatch):
    """The lock targets the construction-time path even if the caller changes cwd."""
    monkeypatch.chdir(tmp_path)
    lock = InstanceLock(Path("photoaident.lock"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    try:
        assert lock.acquire()
        assert (tmp_path / "photoaident.lock").exists()
        assert not (elsewhere / "photoaident.lock").exists()
    finally:
        lock.release()


def test_existing_lock_read_error_fails_closed(tmp_path, caplog):
    """Unreadable metadata cannot trigger automatic deletion."""
    lock = InstanceLock(tmp_path / "photoaident.lock")
    with patch.object(Path, "open", side_effect=PermissionError("denied")):
        assert not lock.acquire()
    assert "Cannot read instance lock" in caplog.text
