import inspect
import threading
from pathlib import Path
from typing import cast, Any

import faiss
import numpy as np
import pytest

from photoaident.db.vector_store import VectorStore

_rng = np.random.default_rng(seed=42)


def test_vector_store_add_search(vector_store):
    # Create some dummy embeddings
    # FAISS works best with normalized vectors for IndexFlatIP
    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    v1 /= np.linalg.norm(v1)

    v2 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    v2 /= np.linalg.norm(v2)

    vector_store.add(1, v1)
    vector_store.add(2, v2)

    # Search for v1
    results = vector_store.search(v1, k=5)
    assert len(results) == 2
    assert results[0][0] == 1
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vector_store_threshold(vector_store):
    v1 = np.zeros(VectorStore.DEFAULT_DIMENSION, dtype=VectorStore.EMBEDDING_DTYPE)
    v1[0] = 1.0

    v2 = np.zeros(VectorStore.DEFAULT_DIMENSION, dtype=VectorStore.EMBEDDING_DTYPE)
    v2[1] = 1.0

    vector_store.add(1, v1)
    vector_store.add(2, v2)

    # Search with v1, threshold 0.5
    results = vector_store.search(v1, k=5, threshold=0.5)
    assert len(results) == 1
    assert results[0][0] == 1


def test_vector_store_get_embedding(vector_store):
    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    v1 /= np.linalg.norm(v1)

    vector_store.add(10, v1)
    retrieved = vector_store.get_embedding(10)

    assert np.allclose(v1, retrieved, atol=1e-5)


def test_vector_store_save_load(vector_store, tmp_path):
    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    v1 /= np.linalg.norm(v1)
    vector_store.add(1, v1)

    index_path = tmp_path / "test.index"
    vector_store.save(index_path)

    new_store = VectorStore()
    new_store.load(index_path)

    assert new_store.index.ntotal == 1
    retrieved = new_store.get_embedding(1)
    assert np.allclose(v1, retrieved, atol=1e-5)


def test_vector_store_empty_search(vector_store):
    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    results = vector_store.search(v1, k=5)
    assert results == []


def test_vector_store_invalid_id(vector_store):
    with pytest.raises(IndexError):
        vector_store.get_embedding(0)

    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    vector_store.add(1, v1)

    with pytest.raises(IndexError):
        vector_store.get_embedding(999)


def test_vector_store_invalid_dimension(vector_store):
    v_invalid = _rng.random(256).astype(VectorStore.EMBEDDING_DTYPE)
    with pytest.raises(
        ValueError, match=f"must be {VectorStore.DEFAULT_DIMENSION}-dimensional"
    ):
        vector_store.add(1, v_invalid)


def test_vector_store_search_invalid_dimension(vector_store):
    v_invalid = _rng.random(256).astype(VectorStore.EMBEDDING_DTYPE)
    with pytest.raises(
        ValueError, match=f"must be {VectorStore.DEFAULT_DIMENSION}-dimensional"
    ):
        vector_store.search(v_invalid, k=5)


def test_vector_store_search_k0(vector_store):
    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    vector_store.add(1, v1)
    results = vector_store.search(v1, k=0)
    assert results == []


def test_vector_store_load_nonexistent(vector_store):
    with pytest.raises(FileNotFoundError, match="Index file not found"):
        vector_store.load(Path("nonexistent.index"))


def test_vector_store_load_wrong_dimension(tmp_path):
    # Create an index with dimension 256
    index_256 = getattr(faiss, "IndexFlatIP")(256)
    index_path = tmp_path / "index_256.index"
    getattr(faiss, "write_index")(index_256, str(index_path))

    store = VectorStore(dimension=VectorStore.DEFAULT_DIMENSION)
    expected_dim = VectorStore.DEFAULT_DIMENSION
    with pytest.raises(
        ValueError,
        match=f"Loaded index dimension 256 does not match {expected_dim}",
    ):
        store.load(index_path)


def test_all_public_methods_are_locked():
    """All public methods (except __init__) must use the @_locked decorator."""
    public_methods = [
        name
        for name, _ in inspect.getmembers(VectorStore, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    assert len(public_methods) > 0, "Expected at least one public method"
    for name in public_methods:
        method = getattr(VectorStore, name)
        assert hasattr(
            method, "__wrapped__"
        ), f"VectorStore.{name} is missing the @_locked decorator"


class _InstrumentedLock:
    """Drop-in replacement for ``threading.Lock`` that records peak concurrent holders.

    Wraps a real ``threading.Lock`` so the test exercises actual serialization.
    After each successful ``acquire()``, a separate ``_guard`` lock protects an
    atomic bump of ``_holders``; ``max_holders`` captures the peak.  With the
    real lock in place, ``max_holders`` should always be 1 — proving that all
    ``VectorStore`` methods are properly serialized.
    """

    def __init__(self):
        self._real = threading.Lock()
        self._guard = threading.Lock()
        self._holders = 0
        self.max_holders = 0

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()

    def acquire(self, *args, **kwargs):
        result = self._real.acquire(*args, **kwargs)
        if result:
            with self._guard:
                self._holders += 1
                self.max_holders = max(self.max_holders, self._holders)
        return result

    def release(self):
        with self._guard:
            self._holders -= 1
        self._real.release()


# ===========================================================================
# remove
# ===========================================================================


def test_remove_deletes_embedding(vector_store):
    """Remove an added embedding so search no longer finds it."""
    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    v1 /= np.linalg.norm(v1)

    vector_store.add(10, v1)
    assert vector_store.index.ntotal == 1

    vector_store.remove(10)
    assert vector_store.index.ntotal == 0

    results = vector_store.search(v1, k=5)
    assert results == []


def test_remove_only_target_embedding(vector_store):
    """Remove one embedding while leaving others intact."""
    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    v1 /= np.linalg.norm(v1)
    v2 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    v2 /= np.linalg.norm(v2)

    vector_store.add(1, v1)
    vector_store.add(2, v2)

    vector_store.remove(1)

    assert vector_store.index.ntotal == 1
    # v2 still retrievable
    retrieved = vector_store.get_embedding(2)
    assert np.allclose(v2, retrieved, atol=1e-5)
    # v1 gone
    with pytest.raises(IndexError):
        vector_store.get_embedding(1)


def test_remove_nonexistent_id_is_noop(vector_store):
    """Removing an ID not in the index does not raise."""
    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    v1 /= np.linalg.norm(v1)
    vector_store.add(1, v1)

    # Should not raise
    vector_store.remove(999)
    assert vector_store.index.ntotal == 1


# ===========================================================================
# needs_migration
# ===========================================================================


def test_needs_migration_false_for_fresh_store(vector_store):
    """A freshly created VectorStore does not need migration."""
    assert vector_store.needs_migration() is False


def test_needs_migration_true_for_bare_index():
    """A store with a bare IndexFlatIP (old format) needs migration."""
    store = VectorStore()
    store.index = faiss.IndexFlatIP(VectorStore.DEFAULT_DIMENSION)
    assert store.needs_migration() is True


def test_needs_migration_false_after_save_load(vector_store, tmp_path):
    """An IndexIDMap2 store round-tripped through save/load still reports False."""
    v1 = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(VectorStore.EMBEDDING_DTYPE)
    v1 /= np.linalg.norm(v1)
    vector_store.add(1, v1)

    path = tmp_path / "test.index"
    vector_store.save(path)

    loaded = VectorStore()
    loaded.load(path)
    assert loaded.needs_migration() is False


def test_needs_migration_true_for_loaded_old_format(tmp_path):
    """A bare IndexFlatIP saved to disk and loaded back reports True."""
    raw = faiss.IndexFlatIP(VectorStore.DEFAULT_DIMENSION)
    path = tmp_path / "old.index"
    faiss.write_index(raw, str(path))

    store = VectorStore()
    store.load(path)
    assert store.needs_migration() is True


# ===========================================================================
# concurrent access
# ===========================================================================


def test_concurrent_access_is_serialized(tmp_path):
    """Multiple threads calling VectorStore methods are serialized by the lock."""
    store = VectorStore()

    embedding = (
        np.random.default_rng(0)
        .random(VectorStore.DEFAULT_DIMENSION)
        .astype(VectorStore.EMBEDDING_DTYPE)
    )
    embedding /= np.linalg.norm(embedding)

    # Seed one vector so search/get_embedding have something to hit
    store.add(1, embedding)

    index_path = tmp_path / "concurrent.index"
    store.save(index_path)

    # Replace the lock with our instrumented version
    instrumented_lock = _InstrumentedLock()
    store._lock = cast(Any, instrumented_lock)

    errors: list[str] = []
    errors_lock = threading.Lock()
    barrier = threading.Barrier(5)

    def _worker(fn, label: str):
        try:
            barrier.wait(timeout=5)
            fn()
        except Exception as exc:
            with errors_lock:
                errors.append(f"{label}: {exc}")

    pairs = [
        (lambda: store.add(2, embedding), "add"),
        (lambda: store.search(embedding, k=1), "search"),
        (lambda: store.get_embedding(1), "get_embedding"),
        (lambda: store.save(index_path), "save"),
        (lambda: store.load(index_path), "load"),
    ]

    threads = [
        threading.Thread(target=_worker, args=(fn, label)) for fn, label in pairs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    for t in threads:
        assert not t.is_alive(), "Worker thread did not finish within timeout"

    assert not errors, f"Thread errors: {errors}"
    # The instrumented lock proves all methods acquire it (max_holders == 1
    # because the real lock serializes them). If any method skipped the lock,
    # it wouldn't show up here — the @_locked decorator test covers that.
    assert (
        instrumented_lock.max_holders == 1
    ), f"Lock held by {instrumented_lock.max_holders} threads simultaneously"
