"""Remove ollama_host and ollama_model from classifier_config.

Data already migrated to llm_providers.api_base_url by migration 0002.
This migration copies ollama_model to AppSettings KV (ocr_model key)
and then drops the columns from classifier_config.

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Step 1: Copy ollama_model to AppSettings KV store as ocr_model
    # (ollama_host was already copied to llm_providers.api_base_url by migration 0002)
    try:
        row = conn.execute(
            sa.text("SELECT ollama_model FROM classifier_config WHERE id = 1")
        ).fetchone()
        if row and row[0]:
            # Upsert into app_settings KV store
            existing = conn.execute(
                sa.text("SELECT id FROM app_settings WHERE key = 'ocr_model'")
            ).fetchone()
            if existing:
                conn.execute(
                    sa.text("UPDATE app_settings SET value = :v WHERE key = 'ocr_model'"),
                    {"v": row[0]}
                )
            else:
                conn.execute(
                    sa.text(
                        "INSERT INTO app_settings (key, value, value_type) VALUES ('ocr_model', :v, 'str')"
                    ),
                    {"v": row[0]}
                )
    except Exception:
        pass  # Column may not exist in fresh databases

    # Step 2: Drop ollama_host and ollama_model columns
    with op.batch_alter_table("classifier_config") as batch_op:
        batch_op.drop_column("ollama_host")
        batch_op.drop_column("ollama_model")


def downgrade() -> None:
    # Data already migrated — downgrade adds back nullable columns as a best-effort
    with op.batch_alter_table("classifier_config") as batch_op:
        batch_op.add_column(
            sa.Column("ollama_host", sa.String(500), nullable=True, server_default=None)
        )
        batch_op.add_column(
            sa.Column("ollama_model", sa.String(200), nullable=True, server_default=None)
        )
