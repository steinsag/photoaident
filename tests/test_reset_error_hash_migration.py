"""Tests for the f6a7b8c9d0e1 migration (reset ERROR file hashes to NULL)."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session


def _alembic_cfg(db_url: str) -> Config:
    """Build an Alembic Config pointing at the project's migrations directory."""
    migrations_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "photoaident"
        / "db"
        / "migrations"
    )
    cfg = Config()
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


_PREV_REV = "e5f6a7b8c9d0"
_TARGET_REV = "f6a7b8c9d0e1"


# ===========================================================================
# upgrade — ERROR hashes reset to NULL
# ===========================================================================


def test_error_hash_reset_to_null(tmp_path):
    """Images with file_hash='ERROR' are set to NULL after migration."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    cfg = _alembic_cfg(db_url)

    # Apply migrations up to the revision before the new one
    command.upgrade(cfg, _PREV_REV)

    engine = create_engine(db_url)
    with Session(engine) as session:
        session.execute(
            text(
                "INSERT INTO images (file_path, file_hash, file_size, index_version) "
                "VALUES (:path, :hash, :size, 1)"
            ),
            [
                {"path": "/img/error1.jpg", "hash": "ERROR", "size": 100},
                {"path": "/img/error2.jpg", "hash": "ERROR", "size": 200},
            ],
        )
        session.commit()

    # Now apply the target migration
    command.upgrade(cfg, _TARGET_REV)

    with Session(engine) as session:
        rows = session.execute(
            text("SELECT file_path, file_hash FROM images ORDER BY file_path")
        ).all()

    assert len(rows) == 2
    for row in rows:
        assert row[1] is None, f"Expected NULL hash for {row[0]}, got {row[1]!r}"

    engine.dispose()


# ===========================================================================
# upgrade — other hash values are NOT affected
# ===========================================================================


def test_normal_hash_not_affected(tmp_path):
    """Images with a normal file_hash are unchanged after migration."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    cfg = _alembic_cfg(db_url)
    command.upgrade(cfg, _PREV_REV)

    engine = create_engine(db_url)
    with Session(engine) as session:
        session.execute(
            text(
                "INSERT INTO images (file_path, file_hash, file_size, index_version) "
                "VALUES (:path, :hash, :size, 1)"
            ),
            {"path": "/img/normal.jpg", "hash": "abc123def456", "size": 300},
        )
        session.commit()

    command.upgrade(cfg, _TARGET_REV)

    with Session(engine) as session:
        row = session.execute(
            text("SELECT file_hash FROM images WHERE file_path = '/img/normal.jpg'")
        ).one()

    assert row[0] == "abc123def456"
    engine.dispose()


def test_missing_hash_not_affected(tmp_path):
    """Images with file_hash='MISSING' are unchanged after migration."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    cfg = _alembic_cfg(db_url)
    command.upgrade(cfg, _PREV_REV)

    engine = create_engine(db_url)
    with Session(engine) as session:
        session.execute(
            text(
                "INSERT INTO images (file_path, file_hash, file_size, index_version) "
                "VALUES (:path, :hash, :size, 1)"
            ),
            {"path": "/img/missing.jpg", "hash": "MISSING", "size": 400},
        )
        session.commit()

    command.upgrade(cfg, _TARGET_REV)

    with Session(engine) as session:
        row = session.execute(
            text("SELECT file_hash FROM images WHERE file_path = '/img/missing.jpg'")
        ).one()

    assert row[0] == "MISSING"
    engine.dispose()


def test_null_hash_not_affected(tmp_path):
    """Images with file_hash=NULL remain NULL after migration."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    cfg = _alembic_cfg(db_url)
    command.upgrade(cfg, _PREV_REV)

    engine = create_engine(db_url)
    with Session(engine) as session:
        session.execute(
            text(
                "INSERT INTO images (file_path, file_hash, file_size, index_version) "
                "VALUES (:path, NULL, :size, 1)"
            ),
            {"path": "/img/null.jpg", "size": 500},
        )
        session.commit()

    command.upgrade(cfg, _TARGET_REV)

    with Session(engine) as session:
        row = session.execute(
            text("SELECT file_hash FROM images WHERE file_path = '/img/null.jpg'")
        ).one()

    assert row[0] is None
    engine.dispose()


# ===========================================================================
# upgrade — mixed data
# ===========================================================================


def test_mixed_hashes_only_error_reset(tmp_path):
    """Only ERROR hashes are reset; MISSING, NULL, and normal hashes survive."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    cfg = _alembic_cfg(db_url)
    command.upgrade(cfg, _PREV_REV)

    engine = create_engine(db_url)
    with Session(engine) as session:
        session.execute(
            text(
                "INSERT INTO images (file_path, file_hash, file_size, index_version) "
                "VALUES (:path, :hash, :size, 1)"
            ),
            [
                {"path": "/a.jpg", "hash": "ERROR", "size": 10},
                {"path": "/b.jpg", "hash": "MISSING", "size": 20},
                {"path": "/c.jpg", "hash": None, "size": 30},
                {"path": "/d.jpg", "hash": "deadbeef", "size": 40},
            ],
        )
        session.commit()

    command.upgrade(cfg, _TARGET_REV)

    with Session(engine) as session:
        rows = {
            r[0]: r[1]
            for r in session.execute(
                text("SELECT file_path, file_hash FROM images")
            ).all()
        }

    assert rows["/a.jpg"] is None  # ERROR → NULL
    assert rows["/b.jpg"] == "MISSING"
    assert rows["/c.jpg"] is None  # was already NULL
    assert rows["/d.jpg"] == "deadbeef"
    engine.dispose()
