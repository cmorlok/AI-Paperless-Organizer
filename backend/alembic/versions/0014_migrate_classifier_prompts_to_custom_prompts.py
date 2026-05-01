"""Migrate classifier_config prompt columns to custom_prompts table.

Moves prompt_title, prompt_tags, prompt_correspondent, prompt_document_type,
prompt_date from classifier_config to custom_prompts, then drops the columns.

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-30
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


FIELD_MAPPING = [
    ("prompt_title", "classifier_rules_title"),
    ("prompt_tags", "classifier_rules_tags"),
    ("prompt_correspondent", "classifier_rules_correspondent"),
    ("prompt_document_type", "classifier_rules_doctype"),
    ("prompt_date", "classifier_rules_date"),
]


def upgrade() -> None:
    conn = op.get_bind()

    for old_col, new_key in FIELD_MAPPING:
        row = conn.execute(
            sa.text(f"SELECT {old_col} FROM classifier_config WHERE id = 1")
        ).fetchone()

        if row and row[0]:
            value = row[0]
            if value and value.strip():
                existing = conn.execute(
                    sa.text("SELECT id FROM custom_prompts WHERE entity_type = :key"),
                    {"key": new_key}
                ).fetchone()

                if existing:
                    conn.execute(
                        sa.text("""
                            UPDATE custom_prompts
                            SET prompt_template = :value, is_active = 1, modified = 1
                            WHERE entity_type = :key
                        """),
                        {"key": new_key, "value": value}
                    )
                else:
                    conn.execute(
                        sa.text("""
                            INSERT INTO custom_prompts (entity_type, prompt_template, is_active, modified)
                            VALUES (:key, :value, 1, 1)
                        """),
                        {"key": new_key, "value": value}
                    )

    for old_col, _ in FIELD_MAPPING:
        op.drop_column("classifier_config", old_col)


def downgrade() -> None:
    for old_col, new_key in reversed(FIELD_MAPPING):
        op.add_column(
            "classifier_config",
            sa.Column(old_col, sa.Text(), nullable=True)
        )

    conn = op.get_bind()

    for old_col, new_key in reversed(FIELD_MAPPING):
        row = conn.execute(
            sa.text("SELECT prompt_template FROM custom_prompts WHERE entity_type = :key"),
            {"key": new_key}
        ).fetchone()

        if row:
            conn.execute(
                sa.text(f"UPDATE classifier_config SET {old_col} = :value WHERE id = 1"),
                {"value": row[0]}
            )
            conn.execute(
                sa.text("DELETE FROM custom_prompts WHERE entity_type = :key"),
                {"key": new_key}
            )
