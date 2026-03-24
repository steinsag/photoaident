"""Tests for migration f6a7b8c9d0e1 — reset ERROR images for re-indexing."""

from sqlalchemy import select, text

from photoaident.db.database import Image


def test_migration_resets_error_images_to_null(search_db):
    """upgrade() sets file_hash = NULL for all images with file_hash = 'ERROR'."""
    with search_db() as session:
        session.add(Image(file_path="/a.jpg", file_size=100, file_hash="ERROR"))
        session.add(Image(file_path="/b.jpg", file_size=100, file_hash="ERROR"))
        session.commit()

    with search_db() as session:
        session.execute(
            text("UPDATE images SET file_hash = NULL WHERE file_hash = 'ERROR'")
        )
        session.commit()

    with search_db() as session:
        rows = session.execute(select(Image)).scalars().all()
    assert all(img.file_hash is None for img in rows)


def test_migration_leaves_other_hashes_untouched(search_db):
    """upgrade() does not touch images with file_hash values other than 'ERROR'."""
    with search_db() as session:
        session.add(Image(file_path="/ok.jpg", file_size=100, file_hash="abc123"))
        session.add(Image(file_path="/miss.jpg", file_size=100, file_hash="MISSING"))
        session.add(Image(file_path="/null.jpg", file_size=100, file_hash=None))
        session.commit()

    with search_db() as session:
        session.execute(
            text("UPDATE images SET file_hash = NULL WHERE file_hash = 'ERROR'")
        )
        session.commit()

    with search_db() as session:
        rows = {
            img.file_path: img.file_hash
            for img in session.execute(select(Image)).scalars().all()
        }
    assert rows["/ok.jpg"] == "abc123"
    assert rows["/miss.jpg"] == "MISSING"
    assert rows["/null.jpg"] is None


def test_migration_is_idempotent_on_empty_table(search_db):
    """upgrade() runs without error when no ERROR images exist."""
    with search_db() as session:
        session.execute(
            text("UPDATE images SET file_hash = NULL WHERE file_hash = 'ERROR'")
        )
        session.commit()
    # No exception → pass
