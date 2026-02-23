from pathlib import Path

import faiss
import numpy as np
import pytest

from photoaident.db.vector_store import VectorStore

_rng = np.random.default_rng(seed=42)


def test_vector_store_add_search(vector_store):
    # Create some dummy embeddings
    # FAISS works best with normalized vectors for IndexFlatIP
    v1 = _rng.random(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    v2 = _rng.random(512).astype(np.float32)
    v2 /= np.linalg.norm(v2)

    id1 = vector_store.add(v1)
    id2 = vector_store.add(v2)

    assert id1 == 0
    assert id2 == 1

    # Search for v1
    results = vector_store.search(v1, k=5)
    assert len(results) == 2
    assert results[0][0] == id1
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vector_store_threshold(vector_store):
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0

    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0

    vector_store.add(v1)
    vector_store.add(v2)

    # Search with v1, threshold 0.5
    results = vector_store.search(v1, k=5, threshold=0.5)
    assert len(results) == 1
    assert results[0][0] == 0


def test_vector_store_get_embedding(vector_store):
    v1 = _rng.random(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    faiss_id = vector_store.add(v1)
    retrieved = vector_store.get_embedding(faiss_id)

    assert np.allclose(v1, retrieved, atol=1e-5)


def test_vector_store_save_load(vector_store, tmp_path):
    v1 = _rng.random(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    vector_store.add(v1)

    index_path = tmp_path / "test.index"
    vector_store.save(index_path)

    new_store = VectorStore()
    new_store.load(index_path)

    assert new_store.index.ntotal == 1
    retrieved = new_store.get_embedding(0)
    assert np.allclose(v1, retrieved, atol=1e-5)


def test_vector_store_empty_search(vector_store):
    v1 = _rng.random(512).astype(np.float32)
    results = vector_store.search(v1, k=5)
    assert results == []


def test_vector_store_invalid_id(vector_store):
    with pytest.raises(IndexError):
        vector_store.get_embedding(0)

    v1 = _rng.random(512).astype(np.float32)
    vector_store.add(v1)

    with pytest.raises(IndexError):
        vector_store.get_embedding(1)


def test_vector_store_invalid_dimension(vector_store):
    v_invalid = _rng.random(256).astype(np.float32)
    with pytest.raises(ValueError, match="must be 512-dimensional"):
        vector_store.add(v_invalid)


def test_vector_store_search_k0(vector_store):
    v1 = _rng.random(512).astype(np.float32)
    vector_store.add(v1)
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

    store = VectorStore(dimension=512)
    with pytest.raises(
        ValueError, match="Loaded index dimension 256 does not match 512"
    ):
        store.load(index_path)
