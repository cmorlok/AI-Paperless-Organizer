"""create auth_config and drop password columns

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create auth_config table
    op.create_table(
        'auth_config',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('password_hash', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    # Drop password columns from app_settings (legacy SHA-256 hashes are discarded)
    with op.batch_alter_table('app_settings') as batch_op:
        batch_op.drop_column('password_hash')
        batch_op.drop_column('password_enabled')


def downgrade() -> None:
    # Add columns back
    with op.batch_alter_table('app_settings') as batch_op:
        batch_op.add_column(sa.Column('password_enabled', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('password_hash', sa.String(length=500), nullable=True))
    op.drop_table('auth_config')
