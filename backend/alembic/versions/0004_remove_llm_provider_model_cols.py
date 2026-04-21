"""Remove model/classifier_model/vision_model from LlmProvider (LLM-09, D-05).

LLMProvider is now connection config only. Job model routing is via AppSettings key-value.

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the model columns — LLMProvider is now connection config only (D-05)
    op.drop_column("llm_providers", "model")
    op.drop_column("llm_providers", "classifier_model")
    op.drop_column("llm_providers", "vision_model")


def downgrade() -> None:
    op.add_column(
        "llm_providers",
        sa.Column("model", sa.String(200), server_default="", nullable=True),
    )
    op.add_column(
        "llm_providers",
        sa.Column("classifier_model", sa.String(200), server_default="", nullable=True),
    )
    op.add_column(
        "llm_providers",
        sa.Column("vision_model", sa.String(200), server_default="", nullable=True),
    )
