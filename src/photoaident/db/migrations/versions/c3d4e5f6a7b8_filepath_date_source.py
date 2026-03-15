"""filepath_date_source

Make taken_at_source nullable, migrate filesystem rows to NULL, and replace
the "filesystem" enum value with "filepath".

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-15 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_OLD_ENUM = sa.Enum("exif", "filesystem", "manual", name="takenatsource")
_NEW_ENUM = sa.Enum("exif", "filepath", "manual", name="takenatsource")


def upgrade() -> None:
    # Rows whose date came from the unreliable filesystem mtime are cleared:
    # set both taken_at and taken_at_source to NULL before the constraint changes.
    op.execute(
        "UPDATE image_metadata SET taken_at = NULL, taken_at_source = NULL "
        "WHERE taken_at_source = 'filesystem'"
    )

    # SQLite requires batch mode to alter column constraints.  The type change
    # from the old enum (includes 'filesystem') to the new one (includes
    # 'filepath') updates the CHECK constraint when SQLite recreates the table.
    with op.batch_alter_table("image_metadata") as batch_op:
        batch_op.alter_column(
            "taken_at_source",
            existing_type=_OLD_ENUM,
            type_=_NEW_ENUM,
            nullable=True,
        )


def downgrade() -> None:
    # 'filepath' is not in the old enum; remap to 'filesystem' before the type change.
    op.execute(
        "UPDATE image_metadata SET taken_at_source = 'filesystem' "
        "WHERE taken_at_source = 'filepath'"
    )
    # Restore taken_at_source to NOT NULL, setting any NULLs back to 'filesystem'
    # so that existing rows are valid after the column becomes non-nullable again.
    op.execute(
        "UPDATE image_metadata SET taken_at_source = 'filesystem' "
        "WHERE taken_at_source IS NULL"
    )
    with op.batch_alter_table("image_metadata") as batch_op:
        batch_op.alter_column(
            "taken_at_source",
            existing_type=_NEW_ENUM,
            type_=_OLD_ENUM,
            nullable=False,
        )
