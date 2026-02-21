import pytest

from photoaident.db.database import get_engine
from photoaident.db.migrate import apply_migrations
from photoaident.paths import AppPaths


@pytest.fixture(scope="session")
def tmp_paths(tmp_path_factory) -> AppPaths:
    """Isolated XDG paths for the test session — never touches real user data."""
    base = tmp_path_factory.mktemp("photoaident")
    paths = AppPaths(
        base_data=base / "data",
        base_cache=base / "cache",
        base_config=base / "config",
    )
    paths.ensure_dirs()
    return paths


@pytest.fixture(scope="session")
def db_engine(tmp_paths):
    """SQLite engine with all Alembic migrations applied."""
    engine = get_engine(str(tmp_paths.db_path))
    apply_migrations(f"sqlite:///{tmp_paths.db_path}")
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


@pytest.fixture(scope="session")
def vector_store(tmp_paths):
    from photoaident.db.vector_store import VectorStore

    return VectorStore(tmp_paths.faiss_path)
