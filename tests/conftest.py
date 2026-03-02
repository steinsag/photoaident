from typing import Generator

import pytest

from photoaident.db.database import get_engine
from photoaident.db.migrate import apply_migrations
from photoaident.paths import AppPaths


@pytest.fixture
def tmp_app_paths(tmp_path_factory) -> Generator[AppPaths, None, None]:
    """Isolated XDG paths per test — never touches real user data."""
    base = tmp_path_factory.mktemp("photoaident")
    AppPaths._data_override = base / "data"
    AppPaths._cache_override = base / "cache"
    AppPaths._config_override = base / "config"
    paths = AppPaths()
    paths.ensure_dirs()
    yield paths
    AppPaths._data_override = None
    AppPaths._cache_override = None
    AppPaths._config_override = None


@pytest.fixture
def db_engine(tmp_app_paths):
    """SQLite engine with all Alembic migrations applied."""
    engine = get_engine(str(tmp_app_paths.db_path))
    apply_migrations(f"sqlite:///{tmp_app_paths.db_path}")
    return engine


@pytest.fixture
def db_session(db_engine):
    """Per-test transactional session — rolls back after each test."""
    from sqlalchemy.orm import Session

    with db_engine.connect() as conn:
        with conn.begin_nested() as savepoint:
            session = Session(bind=conn)
            yield session
            session.close()
            savepoint.rollback()


@pytest.fixture
def vector_store():
    from photoaident.db.vector_store import VectorStore

    return VectorStore()


@pytest.fixture(scope="session")
def face_embedder():
    """A single FaceEmbedder instance shared across the whole test session.

    FaceEmbedder initialisation loads the ONNX model from disk (~3-4 s).
    Sharing the instance avoids paying that cost once per test.
    """
    from photoaident.core.embeddings import FaceEmbedder

    return FaceEmbedder(ctx_id=-1)
