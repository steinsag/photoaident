from unittest.mock import patch

from photoaident.utils.instance_lock import InstanceLock


def test_instance_lock_acquire_release(tmp_path):
    lock_path = tmp_path / "photoaident.lock"
    lock = InstanceLock(lock_path)

    # First acquire should succeed
    assert lock.acquire() is True
    assert lock_path.exists()

    # Second acquire should fail
    lock2 = InstanceLock(lock_path)
    assert lock2.acquire() is False

    # Release first lock
    lock.release()

    # Now second acquire should succeed
    assert lock2.acquire() is True
    lock2.release()
    assert not lock_path.exists()


def test_instance_lock_context_independence(tmp_path):
    lock_path = tmp_path / "photoaident.lock"
    lock1 = InstanceLock(lock_path)
    lock2 = InstanceLock(lock_path)

    assert lock1.acquire() is True
    assert lock2.acquire() is False

    lock1.release()
    assert lock2.acquire() is True
    lock2.release()


def test_release_handles_flock_error(tmp_path):
    """release() silently swallows IOError from fcntl.flock."""
    lock_path = tmp_path / "photoaident.lock"
    lock = InstanceLock(lock_path)
    assert lock.acquire() is True

    with patch(
        "photoaident.utils.instance_lock.fcntl.flock",
        side_effect=IOError("flock failed"),
    ):
        lock.release()  # must not raise

    # Lock file should be cleaned up despite the flock error
    assert not lock_path.exists()


def test_release_handles_unlink_error(tmp_path):
    """release() silently swallows OSError from lock_path.unlink()."""
    lock_path = tmp_path / "photoaident.lock"
    lock = InstanceLock(lock_path)
    assert lock.acquire() is True

    with patch("pathlib.Path.unlink", side_effect=OSError("unlink failed")):
        lock.release()  # must not raise

    # _lock_file should be cleared even though unlink failed
    assert lock._lock_file is None
