from pathlib import Path
from typing import Any, List, Tuple

import faiss
import numpy as np


class VectorStore:
    """FAISS wrapper for 512-dimensional face embeddings using IndexFlatIP.

    IndexFlatIP uses Inner Product similarity, which for L2-normalized vectors
    is equivalent to cosine similarity.
    """

    DIMENSION = 512

    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        # IndexFlatIP does not support IDs by default, but it returns the 0-indexed
        # position which acts as the faiss_id.
        self.index: Any = getattr(faiss, "IndexFlatIP")(self.dimension)

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

        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding must be {self.dimension}-dimensional.")

        # Ensure float32
        embedding = embedding.astype(np.float32)

        # Before adding, get current count
        faiss_id = self.index.ntotal
        self.index.add(embedding)
        return faiss_id

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
        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # FAISS search expects k > 0
        if k <= 0:
            return []

        # Clip k to current index size
        actual_k = min(k, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, actual_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and dist >= threshold:
                results.append((int(idx), float(dist)))

        return results

    def get_embedding(self, faiss_id: int) -> np.ndarray:
        """Retrieves an embedding by its faiss_id.

        Args:
            faiss_id: The ID of the embedding to retrieve.

        Returns:
            The embedding as a numpy array.
        """
        if faiss_id < 0 or faiss_id >= self.index.ntotal:
            raise IndexError(f"faiss_id {faiss_id} is out of bounds.")

        # IndexFlatIP supports reconstruct to get back the vector
        return self.index.reconstruct(faiss_id)

    def reset(self) -> None:
        """Clears all embeddings from the index."""
        self.index.reset()

    def save(self, path: Path) -> None:
        """Saves the FAISS index to a file.

        Args:
            path: Path to the .index file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        getattr(faiss, "write_index")(self.index, str(path))

    def load(self, path: Path) -> None:
        """Loads a FAISS index from a file.

        Args:
            path: Path to the .index file.
        """
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        self.index = getattr(faiss, "read_index")(str(path))
        if self.index.d != self.dimension:
            raise ValueError(
                f"Loaded index dimension {self.index.d} "
                f"does not match {self.dimension}."
            )
