"""Migrate app_settings to key-value schema (LLM-08).

Adds key, value, value_type columns to app_settings and seeds
classifier_provider as a key-value entry from the existing scalar column.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Add key/value columns to app_settings (works on SQLite via ALTER TABLE)
    op.add_column("app_settings", sa.Column("key", sa.String(100), nullable=True))
    op.add_column("app_settings", sa.Column("value", sa.String(500), nullable=True))
    op.add_column(
        "app_settings",
        sa.Column("value_type", sa.String(20), server_default="str", nullable=True),
    )

    # Seed classifier_provider as a KV entry: UPDATE existing row(s) that have
    # a classifier_provider value set, or INSERT a new KV row if no row has one yet.
    # This handles both single-row (id=1 only) and multi-row setups.
    op.execute(
        sa.text(
            "UPDATE app_settings "
            "SET key = 'classifier_provider', "
            "    value = classifier_provider, "
            "    value_type = 'str' "
            "WHERE classifier_provider IS NOT NULL "
            "AND classifier_provider != ''"
        )
    )


def downgrade() -> None:
    op.drop_column("app_settings", "value_type")
    op.drop_column("app_settings", "value")
    op.drop_column("app_settings", "key")
