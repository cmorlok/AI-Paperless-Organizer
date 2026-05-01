"""Migrate custom field prompts to custom_prompts table.

Moves extraction_prompt from classifier_custom_field_mappings to custom_prompts,
then drops the column. Also drops validation_regex (dead code).

Revision ID: 0015
Revises: 0014
Create Date: 2026-05-01
"""

import re
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name

    # Widen entity_type column to allow longer keys
    if dialect == "sqlite":
        # SQLite: recreate table to change column type
        conn.execute(sa.text("""
            CREATE TABLE custom_prompts_new (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                entity_type VARCHAR(100) NOT NULL,
                prompt_template TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                modified BOOLEAN NOT NULL DEFAULT 0,
                created_at DATETIME,
                updated_at DATETIME
            )
        """))
        conn.execute(sa.text("""
            INSERT INTO custom_prompts_new (id, entity_type, prompt_template, is_active, modified, created_at, updated_at)
            SELECT id, entity_type, prompt_template, is_active, modified, created_at, updated_at FROM custom_prompts
        """))
        conn.execute(sa.text("DROP TABLE custom_prompts"))
        conn.execute(sa.text("ALTER TABLE custom_prompts_new RENAME TO custom_prompts"))
    else:
        op.alter_column("custom_prompts", "entity_type", existing_type=sa.String(50), type_=sa.String(100))

    # Migrate user-defined extraction_prompts to custom_prompts
    rows = conn.execute(
        sa.text("""
            SELECT paperless_field_id, paperless_field_name, extraction_prompt
            FROM classifier_custom_field_mappings
            WHERE extraction_prompt IS NOT NULL AND extraction_prompt != ''
        """)
    ).fetchall()

    for paperless_field_id, paperless_field_name, extraction_prompt in rows:
        slugified_name = re.sub(r"[^a-z0-9]+", "", paperless_field_name.lower())
        prompt_key = f"classifier_rules_custom_fields_{slugified_name}"
        existing = conn.execute(
            sa.text("SELECT id FROM custom_prompts WHERE entity_type = :key"),
            {"key": prompt_key}
        ).fetchone()

        if existing:
            conn.execute(
                sa.text("""
                    UPDATE custom_prompts
                    SET prompt_template = :value, is_active = 1, modified = 1
                    WHERE entity_type = :key
                """),
                {"key": prompt_key, "value": extraction_prompt}
            )
        else:
            conn.execute(
                sa.text("""
                    INSERT INTO custom_prompts (entity_type, prompt_template, is_active, modified)
                    VALUES (:key, :value, 1, 1)
                """),
                {"key": prompt_key, "value": extraction_prompt}
            )

    # Drop extraction_prompt and validation_regex columns
    op.drop_column("classifier_custom_field_mappings", "extraction_prompt")
    op.drop_column("classifier_custom_field_mappings", "validation_regex")


def downgrade() -> None:
    op.add_column(
        "classifier_custom_field_mappings",
        sa.Column("extraction_prompt", sa.Text(), nullable=True)
    )
    op.add_column(
        "classifier_custom_field_mappings",
        sa.Column("validation_regex", sa.String(500), nullable=True)
    )

    conn = op.get_bind()

    # Migrate prompts back from custom_prompts to classifier_custom_field_mappings
    rows = conn.execute(
        sa.text("""
            SELECT prompt_template, entity_type
            FROM custom_prompts
            WHERE entity_type LIKE 'classifier_rules_custom_fields_%'
        """)
    ).fetchall()

    for prompt_template, entity_type in rows:
        match = entity_type.replace("classifier_rules_custom_fields_", "")
        conn.execute(
            sa.text("""
                UPDATE classifier_custom_field_mappings
                SET extraction_prompt = :value
                WHERE LOWER(REPLACE(paperless_field_name, ' ', '')) = :match
                   OR LOWER(REPLACE(paperless_field_name, '-', '')) = :match
            """),
            {"value": prompt_template, "match": match}
        )

    conn.execute(
        sa.text("""
            DELETE FROM custom_prompts
            WHERE entity_type LIKE 'classifier_rules_custom_fields_%'
        """)
    )

    dialect = conn.dialect.name
    if dialect == "sqlite":
        conn.execute(sa.text("""
            CREATE TABLE custom_prompts_new (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                entity_type VARCHAR(50) NOT NULL,
                prompt_template TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                modified BOOLEAN NOT NULL DEFAULT 0,
                created_at DATETIME,
                updated_at DATETIME
            )
        """))
        conn.execute(sa.text("""
            INSERT INTO custom_prompts_new (id, entity_type, prompt_template, is_active, modified, created_at, updated_at)
            SELECT id, entity_type, prompt_template, is_active, modified, created_at, updated_at FROM custom_prompts
        """))
        conn.execute(sa.text("DROP TABLE custom_prompts"))
        conn.execute(sa.text("ALTER TABLE custom_prompts_new RENAME TO custom_prompts"))