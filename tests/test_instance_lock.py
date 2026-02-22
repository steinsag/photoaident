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
