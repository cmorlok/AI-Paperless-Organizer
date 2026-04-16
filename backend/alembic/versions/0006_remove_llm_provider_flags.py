"""Remove is_active and is_configured from llm_providers.

A provider is implicitly configured if a row exists in llm_providers.
No need for explicit flags.

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('llm_providers', 'is_active')
    op.drop_column('llm_providers', 'is_configured')


def downgrade() -> None:
    op.add_column('llm_providers', sa.Column('is_active', sa.Boolean(), default=False))
    op.add_column('llm_providers', sa.Column('is_configured', sa.Boolean(), default=False))
