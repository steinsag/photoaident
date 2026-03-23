"""make faiss_id nullable

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-03-19 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("faces") as batch_op:
        batch_op.alter_column(
            "faiss_id",
            existing_type=sa.Integer(),
            nullable=True,
        )


def downgrade() -> None:
    # This migration makes faces.faiss_id nullable. After upgrading, new rows
    # may legitimately have faiss_id = NULL. Making the column non-nullable
    # again would either fail if NULLs exist or require inventing placeholder
    # values, and in either case would not restore the original FAISS index
    # positional invariants. To avoid unsafe or lossy downgrades, this
    # migration is explicitly marked as irreversible.
    raise RuntimeError(
        "Downgrade of migration e5f6a7b8c9d0 is not supported: "
        "faces.faiss_id may contain NULL values and FAISS positional "
        "invariants cannot be safely restored."
    )
