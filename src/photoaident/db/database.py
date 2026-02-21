from sqlalchemy import create_engine, Engine


def get_engine(db_path: str | None = None) -> Engine:
    """Creates a SQLAlchemy engine for the SQLite database.

    Args:
        db_path: Path to the database file. If None, uses default.
    """
    if db_path is None:
        from photoaident.paths import AppPaths

        db_path = str(AppPaths().db_path)

    return create_engine(f"sqlite:///{db_path}")
