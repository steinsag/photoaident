from typing import Generator

import pytest
from PySide6.QtCore import QLocale
from sqlalchemy import create_engine

from photoaident.db.database import get_engine, get_session_factory
from photoaident.db.migrate import apply_migrations
from photoaident.db.vector_store import VectorStore
from photoaident.paths import AppPaths


@pytest.fixture
def force_en_us_locale():
    """Force en_US locale for the duration of a test to get stable month names."""
    original = QLocale()
    QLocale.setDefault(QLocale("en_US"))
    yield
    QLocale.setDefault(original)


@pytest.fixture
def tmp_app_paths(tmp_path_factory) -> Generator[AppPaths, None, None]:
    """Isolated XDG paths per test — never touches real user data."""
    base = tmp_path_factory.mktemp("photoaident")
    try:
        AppPaths._data_override = base / "data"
        AppPaths._cache_override = base / "cache"
        AppPaths._config_override = base / "config"
        paths = AppPaths()
        paths.ensure_dirs()
        yield paths
    finally:
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
def vector_store() -> VectorStore:
    """A fresh FAISS VectorStore for tests that need the high-level fixture name."""
    return VectorStore()


@pytest.fixture
def search_db(tmp_path):
    """Fresh per-test SQLite DB with migrations applied (search-layer tests)."""
    db_path = tmp_path / "search.db"
    apply_migrations(f"sqlite:///{db_path}")
    engine = create_engine(f"sqlite:///{db_path}")
    return get_session_factory(engine)


@pytest.fixture(scope="session")
def face_embedder():
    """A single FaceEmbedder instance shared across the whole test session.

    FaceEmbedder initialisation loads the ONNX model from disk (~3-4 s).
    Sharing the instance avoids paying that cost once per test.
    """
    from photoaident.core.embeddings import FaceEmbedder

    return FaceEmbedder(ctx_id=-1)
