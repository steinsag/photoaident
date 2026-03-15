"""add mean_embedding to embedding_clusters

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-03-15 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("embedding_clusters") as batch_op:
        batch_op.add_column(
            sa.Column("mean_embedding", sa.LargeBinary(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("embedding_clusters") as batch_op:
        batch_op.drop_column("mean_embedding")
