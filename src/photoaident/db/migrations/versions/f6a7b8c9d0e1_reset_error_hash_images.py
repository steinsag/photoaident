"""reset error hash images for reindexing

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-03-23 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("UPDATE images SET file_hash = NULL WHERE file_hash = 'ERROR'")


def downgrade() -> None:
    # Data-only migration; cannot recover the original ERROR markers.
    raise RuntimeError(
        "Irreversible migration: cannot restore prior 'ERROR' file_hash markers."
    )
