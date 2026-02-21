from pathlib import Path


class VectorStore:
    def __init__(self, path: Path):
        self.path = path
        # Placeholder for FAISS index initialization
