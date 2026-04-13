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

    # Add key/value columns to app_settings
    op.add_column("app_settings", sa.Column("key", sa.String(100), nullable=True))
    op.add_column("app_settings", sa.Column("value", sa.String(500), nullable=True))
    op.add_column(
        "app_settings",
        sa.Column("value_type", sa.String(20), server_default="str", nullable=True),
    )

    # Make key unique (allows NULL for existing scalar rows)
    op.create_unique_constraint("uq_app_settings_key", "app_settings", ["key"])

    # Seed classifier_provider as a key-value entry from the scalar column
    op.execute(
        sa.text(
            "INSERT INTO app_settings (key, value, value_type) "
            "SELECT 'classifier_provider', classifier_provider, 'str' "
            "FROM app_settings "
            "WHERE classifier_provider IS NOT NULL AND classifier_provider != ''"
        )
    )


def downgrade() -> None:
    op.drop_constraint("uq_app_settings_key", "app_settings", type_="unique")
    op.drop_column("app_settings", "value_type")
    op.drop_column("app_settings", "value")
    op.drop_column("app_settings", "key")