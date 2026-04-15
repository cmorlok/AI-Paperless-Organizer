"""Split AppSettings scalar/KV into distinct rows (LLM-08 fix).

Problem: AppSettings used id as both scalar PK and KV identifier.
The id=1 row has key='classifier_provider' set by migration 0003,
causing SQLAlchemy to treat it as a KV entry and INSERT with duplicate id=1.

Fix:
- id=1, key=NULL → scalar UI settings (password_enabled, etc.)
- id=NULL, key='classifier_provider' → KV entry for classifier_provider value

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Split the single AppSettings row into scalar + KV entry.

    Current state (after 0003):
    - One row with id=1, key='classifier_provider', value=<provider>, and UI columns

    Target state:
    - Scalar row: id=1, key=NULL, UI columns populated, value=NULL
    - KV entry: id=NULL, key='classifier_provider', value=<provider>
    """
    conn = op.get_bind()

    # Step 1: Find the current classifier_provider value from the row
    # (It was stored via 0003 UPDATE which set key='classifier_provider')
    result = conn.execute(
        sa.text("SELECT value FROM app_settings WHERE key = 'classifier_provider'")
    )
    row = result.fetchone()
    classifier_provider_value = row[0] if row else "ollama"

    # Step 2: Reset the id=1 row to be the scalar row (key=NULL, clear KV columns)
    op.execute(
        sa.text(
            "UPDATE app_settings "
            "SET key = NULL, value = NULL, value_type = 'str' "
            "WHERE id = 1"
        )
    )

    # Step 3: Insert a new KV row for classifier_provider
    # (id=NULL allows autoincrement to generate unique id for KV entries)
    op.execute(
        sa.text(
            "INSERT INTO app_settings (id, key, value, value_type) "
            "VALUES (NULL, 'classifier_provider', :cp_value, 'str')"
        ).bindparams(cp_value=classifier_provider_value)
    )

    # Step 4: Add NOT NULL constraint on key column for future KV entries
    # (Scalar row has key=NULL, KV entries have key=<string>)
    # SQLite doesn't support ALTER TABLE ADD CONSTRAINT, so we recreate the table
    with op.batch_alter_table("app_settings", recreate="auto") as batch_op:
        batch_op.alter_column("key", existing_type=sa.String(100), nullable=False)


def downgrade() -> None:
    """Revert: merge KV entry back into scalar row column.

    Merge classifier_provider KV entry back into the scalar row's column.
    """
    conn = op.get_bind()

    # Get classifier_provider value from KV entry
    result = conn.execute(
        sa.text("SELECT value FROM app_settings WHERE key = 'classifier_provider'")
    )
    row = result.fetchone()
    classifier_provider_value = row[0] if row else "ollama"

    # Delete the KV entry
    op.execute(
        sa.text("DELETE FROM app_settings WHERE key = 'classifier_provider'")
    )

    # Restore scalar row with classifier_provider column
    op.execute(
        sa.text(
            "UPDATE app_settings "
            "SET classifier_provider = :cp_value "
            "WHERE id = 1"
        ).bindparams(cp_value=classifier_provider_value)
    )
