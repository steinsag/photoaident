import functools
import threading
from pathlib import Path
from typing import Any, Callable, Concatenate, List, ParamSpec, Tuple, TypeVar

import numpy as np
from faiss import IndexFlatIP, read_index, write_index

FACE_MATCH_THRESHOLD: float = 0.35
"""Minimum cosine similarity for a face embedding match.

Used consistently across person search (library) and face resolution
(image detail dialog) so that search results and face highlights agree.
"""

P = ParamSpec("P")
R = TypeVar("R")


def _locked(
    method: Callable[Concatenate["VectorStore", P], R],
) -> Callable[Concatenate["VectorStore", P], R]:
    """Decorator that acquires self._lock for the duration of the method."""

    @functools.wraps(method)
    def wrapper(self: "VectorStore", *args: P.args, **kwargs: P.kwargs) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class VectorStore:
    """FAISS wrapper for 512-dimensional face embeddings using IndexFlatIP.

    IndexFlatIP uses Inner Product similarity, which for L2-normalized vectors
    is equivalent to cosine similarity.

    All public methods are guarded by a lock so the store can be safely shared
    between the indexing background thread and the UI thread.
    """

    DEFAULT_DIMENSION = 512

    def __init__(self, dimension: int = DEFAULT_DIMENSION):
        self.dimension = dimension
        self._lock = threading.Lock()
        # IndexFlatIP does not support IDs by default, but it returns the 0-indexed
        # position which acts as the faiss_id.
        self.index: Any = IndexFlatIP(self.dimension)

    @_locked
    def add(self, embedding: np.ndarray) -> int:
        """Adds an embedding to the index and returns its faiss_id.

        Args:
            embedding: A 1D or 2D numpy array of shape (512,) or (1, 512).

        Returns:
            The assigned faiss_id (the position in the index).
        """
        # Ensure it's 2D for FAISS
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        elif embedding.ndim != 2:
            raise ValueError(
                f"Embedding must be a 1D or 2D array; got {embedding.ndim}D."
            )

        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding must be {self.dimension}-dimensional.")

        # Ensure float32
        embedding = embedding.astype(np.float32)

        faiss_id = self.index.ntotal
        self.index.add(embedding)
        return faiss_id

    @_locked
    def search(
        self, query_embedding: np.ndarray, k: int, threshold: float = 0.0
    ) -> List[Tuple[int, float]]:
        """Searches for the k most similar embeddings.

        Args:
            query_embedding: A 1D or 2D numpy array of shape (512,) or (1, 512).
            k: Number of neighbors to return.
            threshold: Minimum similarity score to include in results.

        Returns:
            A list of tuples (faiss_id, similarity_score) sorted by similarity.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        elif query_embedding.ndim != 2:
            raise ValueError(
                f"Query embedding must be 1D or 2D; got {query_embedding.ndim}D."
            )

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding must be {self.dimension}-dimensional.")

        query_embedding = query_embedding.astype(np.float32)

        if k <= 0:
            return []

        if self.index.ntotal == 0:
            return []

        actual_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, actual_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and dist >= threshold:
                results.append((int(idx), float(dist)))

        return results

    @_locked
    def get_embedding(self, faiss_id: int) -> np.ndarray:
        """Retrieves an embedding by its faiss_id.

        Args:
            faiss_id: The ID of the embedding to retrieve.

        Returns:
            The embedding as a numpy array.
        """
        if faiss_id < 0 or faiss_id >= self.index.ntotal:
            raise IndexError(f"faiss_id {faiss_id} is out of bounds.")
        return self.index.reconstruct(faiss_id)

    @_locked
    def reset(self) -> None:
        """Clears all embeddings from the index."""
        self.index.reset()

    @_locked
    def save(self, path: Path) -> None:
        """Saves the FAISS index to a file.

        Args:
            path: Path to the .index file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        write_index(self.index, str(path))

    @_locked
    def load(self, path: Path) -> None:
        """Loads a FAISS index from a file.

        Args:
            path: Path to the .index file.
        """
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        loaded = read_index(str(path))
        if loaded.d != self.dimension:
            raise ValueError(
                f"Loaded index dimension {loaded.d} "
                f"does not match {self.dimension}."
            )
        self.index = loaded
