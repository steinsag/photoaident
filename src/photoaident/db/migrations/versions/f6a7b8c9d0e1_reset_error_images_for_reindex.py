"""reset ERROR images for re-indexing

Images whose file_hash was set to 'ERROR' by a failed indexing attempt are
reset to NULL so the next indexing run will retry them automatically.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-03-24 00:00:00.000000

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
    # The original ERROR values are not recoverable; rows that were NULL before
    # this migration are indistinguishable from rows that were reset here.
    raise RuntimeError(
        "Downgrade of migration f6a7b8c9d0e1 is not supported: "
        "Images previously unable to index will be re-processed."
    )
