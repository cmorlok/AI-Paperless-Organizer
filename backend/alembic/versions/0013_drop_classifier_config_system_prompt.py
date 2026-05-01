"""Drop system_prompt column from classifier_config table.

Revision ID: 0013
Revises: 0012
Create Date: 2026-04-30
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("classifier_config", "system_prompt")


def downgrade() -> None:
    op.add_column(
        "classifier_config",
        sa.Column("system_prompt", sa.Text(), nullable=True),
    )
