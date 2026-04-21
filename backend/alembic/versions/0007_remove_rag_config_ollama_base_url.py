"""Remove ollama_base_url from rag_config.

Provider credentials (api_base_url, api_key) are now looked up by llm_service
from the llm_providers table, keyed by provider name. Storing a separate
ollama_base_url in rag_config was redundant.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("rag_config") as batch_op:
        batch_op.drop_column("ollama_base_url")


def downgrade() -> None:
    with op.batch_alter_table("rag_config") as batch_op:
        batch_op.add_column(
            sa.Column("ollama_base_url", sa.String(500), nullable=True,
                      server_default="http://localhost:11434")
        )
