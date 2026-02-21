from pathlib import Path

from alembic import command
from alembic.config import Config


def apply_migrations(db_url: str) -> None:
    """Apply all pending database migrations.

    Args:
        db_url: The SQLAlchemy database URL.
    """
    # Path to alembic.ini relative to this file's location or project root
    # Since we are in src/photoaident/db/migrate.py,
    # alembic.ini is at ../../../alembic.ini
    ini_path = Path(__file__).parent.parent.parent.parent / "alembic.ini"

    cfg = Config(str(ini_path))
    cfg.set_main_option("sqlalchemy.url", db_url)

    # Run the migration
    command.upgrade(cfg, "head")
