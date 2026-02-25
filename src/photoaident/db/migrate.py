from pathlib import Path

from alembic import command
from alembic.config import Config


def apply_migrations(db_url: str) -> None:
    """Apply all pending database migrations.

    Args:
        db_url: The SQLAlchemy database URL.
    """
    # Resolve the migrations directory relative to this file.  This works both
    # in development (src/photoaident/db/migrate.py → .../migrations/) and in a
    # PyInstaller bundle where __file__ points inside sys._MEIPASS and the
    # migrations directory is extracted alongside it as a data tree.
    migrations_dir = Path(__file__).parent / "migrations"

    cfg = Config()
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(cfg, "head")
