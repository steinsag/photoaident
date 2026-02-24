"""add_gps_index_to_image_metadata

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-24 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("idx_metadata_gps", "image_metadata", ["gps_lat", "gps_lon"])


def downgrade() -> None:
    op.drop_index("idx_metadata_gps", table_name="image_metadata")
