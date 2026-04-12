"""Migrate API keys and models from classifier_config into llm_providers.

This is a one-time data migration that was previously run at every startup by
_migrate_classifier_to_providers() in database.py.  It copies API credentials
and model selections from the legacy classifier_config columns into the central
llm_providers table, and propagates the active_provider value into app_settings.

For existing databases created before Alembic was introduced this migration is
NOT run (the database is stamped at head on first startup).  For fresh
installations the classifier_config table will be empty so the migration is a
safe no-op.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-12
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Read the single classifier_config row (id = 1).
    try:
        row = conn.execute(
            sa.text(
                "SELECT active_provider, openai_model, mistral_api_key, mistral_model, "
                "openrouter_api_key, openrouter_model, ollama_host, ollama_model "
                "FROM classifier_config WHERE id = 1"
            )
        ).fetchone()
    except Exception:
        return  # Table doesn't exist yet (shouldn't happen, but be defensive)

    if not row:
        return  # Nothing to migrate

    (
        active_provider,
        openai_model,
        mistral_key,
        mistral_model,
        or_key,
        or_model,
        ollama_host,
        ollama_model,
    ) = row

    # OpenAI: set classifier_model if provided
    if openai_model:
        conn.execute(
            sa.text(
                "UPDATE llm_providers SET classifier_model = :m "
                "WHERE name = 'openai' AND (classifier_model IS NULL OR classifier_model = '')"
            ),
            {"m": openai_model},
        )

    # Mistral: migrate API key and model
    if mistral_key:
        conn.execute(
            sa.text(
                "UPDATE llm_providers SET api_key = :k, classifier_model = :m, is_configured = 1 "
                "WHERE name = 'mistral' AND (api_key IS NULL OR api_key = '')"
            ),
            {"k": mistral_key, "m": mistral_model or "mistral-small-latest"},
        )

    # OpenRouter: migrate API key and model
    if or_key:
        conn.execute(
            sa.text(
                "UPDATE llm_providers SET api_key = :k, classifier_model = :m, is_configured = 1 "
                "WHERE name = 'openrouter' AND (api_key IS NULL OR api_key = '')"
            ),
            {"k": or_key, "m": or_model or "mistralai/mistral-small-2603"},
        )

    # Ollama: set classifier_model and host
    if ollama_model:
        conn.execute(
            sa.text(
                "UPDATE llm_providers SET classifier_model = :m "
                "WHERE name = 'ollama' AND (classifier_model IS NULL OR classifier_model = '')"
            ),
            {"m": ollama_model},
        )
    if ollama_host:
        conn.execute(
            sa.text(
                "UPDATE llm_providers SET api_base_url = :u "
                "WHERE name = 'ollama' "
                "AND (api_base_url IS NULL OR api_base_url = '' "
                "     OR api_base_url = 'http://localhost:11434')"
            ),
            {"u": ollama_host},
        )

    # Propagate active_provider into app_settings
    if active_provider:
        conn.execute(
            sa.text(
                "UPDATE app_settings SET classifier_provider = :p "
                "WHERE id = 1 "
                "AND (classifier_provider IS NULL OR classifier_provider = 'ollama')"
            ),
            {"p": active_provider},
        )


def downgrade() -> None:
    # Data migrations are not reversible without storing the original state.
    pass
