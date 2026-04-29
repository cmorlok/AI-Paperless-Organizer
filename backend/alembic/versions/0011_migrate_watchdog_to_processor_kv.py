"""Migrate ocr_watchdog_* KV keys to ocr_processor_* (D-22 rename).

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    for old_key, new_key in [
        ("ocr_watchdog_enabled", "ocr_processor_enabled"),
        ("ocr_watchdog_interval", "ocr_processor_interval"),
    ]:
        old_row = conn.execute(
            sa.text("SELECT value FROM app_settings WHERE key = :k"),
            {"k": old_key}
        ).fetchone()

        if old_row is None:
            continue

        new_row = conn.execute(
            sa.text("SELECT id FROM app_settings WHERE key = :k"),
            {"k": new_key}
        ).fetchone()

        if new_row is not None:
            continue

        conn.execute(
            sa.text(
                "INSERT INTO app_settings (key, value, value_type) VALUES (:k, :v, 'str')"
            ),
            {"k": new_key, "v": old_row[0]}
        )


def downgrade() -> None:
    pass
