"""add_age_group_to_embedding_clusters

Revision ID: a1b2c3d4e5f6
Revises: 96cc1feaa24c
Create Date: 2026-02-23 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "3971390d31d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add age_group column to embedding_clusters."""
    with op.batch_alter_table("embedding_clusters") as batch_op:
        batch_op.add_column(sa.Column("age_group", sa.String(), nullable=True))


def downgrade() -> None:
    """Remove age_group column from embedding_clusters."""
    with op.batch_alter_table("embedding_clusters") as batch_op:
        batch_op.drop_column("age_group")
