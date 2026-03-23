import functools
import threading
from pathlib import Path
from typing import Any, Callable, Concatenate, List, ParamSpec, Tuple, TypeVar

import numpy as np
from faiss import IndexFlatIP, IndexIDMap2, read_index, write_index

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
    """FAISS wrapper for face embeddings using IndexIDMap2(IndexFlatIP).

    The vector dimensionality is configurable via the ``dimension`` argument
    (defaulting to ``DEFAULT_DIMENSION``).

    IndexFlatIP uses Inner Product similarity, which for L2-normalized vectors
    is equivalent to cosine similarity.  IndexIDMap2 allows storing embeddings
    with database-assigned Face IDs instead of sequential positions.

    All public methods are guarded by a lock so the store can be safely shared
    between the indexing background thread and the UI thread.
    """

    DEFAULT_DIMENSION = 512
    EMBEDDING_DTYPE = np.float32

    def __init__(self, dimension: int = DEFAULT_DIMENSION):
        self.dimension = dimension
        self._lock = threading.Lock()
        self.index: Any = IndexIDMap2(IndexFlatIP(self.dimension))

    def _prepare_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Validate and cast an embedding for FAISS.

        Accepts shape ``(D,)`` or ``(1, D)``, returns a 2-D ``float32`` array
        of shape ``(1, D)``.  Raises ``ValueError`` for wrong ndim or dimension.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        elif embedding.ndim != 2:
            raise ValueError(
                f"Embedding must be a 1D or 2D array; got {embedding.ndim}D."
            )
        # After reshaping/validation above, only a single embedding (1, D) is allowed.
        if embedding.shape[0] != 1:
            raise ValueError(
                f"Batch embeddings with shape (N, D) where N != 1 are not supported; "
                f"got shape {embedding.shape}."
            )
        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding must be {self.dimension}-dimensional.")
        return embedding.astype(VectorStore.EMBEDDING_DTYPE)

    def _require_id_map(self) -> None:
        """Raise RuntimeError if the index has not been migrated to IndexIDMap2."""
        if not isinstance(self.index, IndexIDMap2):
            raise RuntimeError(
                "Index is not an IndexIDMap2 and must be migrated before use. "
                "Call the migration helper to rebuild the index."
            )

    @_locked
    def add(self, face_id: int, embedding: np.ndarray) -> None:
        """Add an embedding to the index with the given face ID.

        Args:
            face_id: The database Face.id to use as the FAISS key.
            embedding: A 1D or 2D numpy array of shape ``(D,)`` or ``(1, D)``,
                where ``D`` is the store's configured ``dimension``.
        """
        self._require_id_map()
        embedding = self._prepare_embedding(embedding)
        ids = np.array([face_id], dtype=np.int64)
        self.index.add_with_ids(embedding, ids)

    @_locked
    def remove(self, face_id: int) -> None:
        """Remove an embedding from the index by its face ID.

        Args:
            face_id: The ID of the embedding to remove.
        """
        self._require_id_map()
        self.index.remove_ids(np.array([face_id], dtype=np.int64))

    @_locked
    def search(
        self, query_embedding: np.ndarray, k: int, threshold: float = 0.0
    ) -> List[Tuple[int, float]]:
        """Search for the k most similar embeddings.

        Args:
            query_embedding: A 1D or 2D numpy array of shape ``(D,)`` or ``(1, D)``,
                where ``D`` is the store's configured ``dimension``.
            k: Number of neighbors to return.
            threshold: Minimum similarity score to include in results.

        Returns:
            A list of tuples (face_id, similarity_score) sorted by similarity.
        """
        self._require_id_map()
        query_embedding = self._prepare_embedding(query_embedding)

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
    def get_embedding(self, face_id: int) -> np.ndarray:
        """Retrieve an embedding by its face ID.

        Args:
            face_id: The ID of the embedding to retrieve.

        Returns:
            The embedding as a numpy array.
        """
        self._require_id_map()
        try:
            return self.index.reconstruct(face_id)
        except RuntimeError as exc:
            raise IndexError(f"face_id {face_id} not found in the index.") from exc

    @_locked
    def reset(self) -> None:
        """Clear all embeddings from the index."""
        self.index.reset()

    @_locked
    def save(self, path: Path) -> None:
        """Save the FAISS index to a file.

        Args:
            path: Path to the .index file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        write_index(self.index, str(path))

    @_locked
    def load(self, path: Path) -> None:
        """Load a FAISS index from a file.

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

    @_locked
    def needs_migration(self) -> bool:
        """Check if the loaded index uses the old format (not IndexIDMap2).

        Returns:
            True if the index needs migration to IndexIDMap2.
        """
        return not isinstance(self.index, IndexIDMap2)
