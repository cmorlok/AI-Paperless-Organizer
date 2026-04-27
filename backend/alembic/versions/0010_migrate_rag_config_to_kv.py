"""Migrate rag_config provider/model to KV.

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    try:
        row = conn.execute(
            sa.text("SELECT chat_model, embedding_model, chat_model_provider, embedding_provider FROM rag_config WHERE id = 1")
        ).fetchone()
    except Exception:
        return

    if not row:
        return

    chat_model, embedding_model, chat_provider, embedding_provider = row

    if chat_model:
        _upsert_kv(conn, "rag_chat_model", chat_model)
        _upsert_kv(conn, "rag_chat_provider", chat_provider or "ollama")
    if embedding_model:
        _upsert_kv(conn, "rag_embedding_model", embedding_model)
        _upsert_kv(conn, "rag_embedding_provider", embedding_provider or "ollama")


def _upsert_kv(conn, key: str, value: str) -> None:
    existing = conn.execute(
        sa.text("SELECT id FROM app_settings WHERE key = :k"),
        {"k": key}
    ).fetchone()
    if existing:
        conn.execute(
            sa.text("UPDATE app_settings SET value = :v WHERE key = :k"),
            {"v": value, "k": key}
        )
    else:
        conn.execute(
            sa.text(
                "INSERT INTO app_settings (key, value, value_type) VALUES (:k, :v, 'str')"
            ),
            {"k": key, "v": value}
        )


def downgrade() -> None:
    pass
