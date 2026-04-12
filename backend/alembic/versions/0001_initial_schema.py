"""Initial database schema.

Creates all tables that were previously managed by the hand-rolled
_migrate_columns() system in database.py.  This represents the complete
schema as it exists after all legacy migrations have been applied.

Revision ID: 0001
Revises:
Create Date: 2026-04-12
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NOW = sa.text("(CURRENT_TIMESTAMP)")


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Settings / providers
    # ------------------------------------------------------------------
    op.create_table(
        "paperless_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("url", sa.String(500), nullable=False),
        sa.Column("api_token", sa.String(500), nullable=False),
        sa.Column("is_configured", sa.Boolean()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "llm_providers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("display_name", sa.String(200), nullable=False),
        sa.Column("api_key", sa.String(500)),
        sa.Column("api_base_url", sa.String(500)),
        sa.Column("model", sa.String(200)),
        sa.Column("classifier_model", sa.String(200)),
        sa.Column("vision_model", sa.String(200)),
        sa.Column("is_active", sa.Boolean()),
        sa.Column("is_configured", sa.Boolean()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "app_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("password_enabled", sa.Boolean()),
        sa.Column("password_hash", sa.String(500)),
        sa.Column("show_debug_menu", sa.Boolean()),
        sa.Column("sidebar_compact", sa.Boolean()),
        sa.Column("classifier_provider", sa.String(100)),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "custom_prompts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("prompt_template", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "ignored_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pattern", sa.String(500), nullable=False),
        sa.Column("reason", sa.String(500)),
        sa.Column("is_regex", sa.Boolean()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "ignored_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("item_name", sa.String(500), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("analysis_type", sa.String(50), nullable=False),
        sa.Column("reason", sa.String(500)),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    # ------------------------------------------------------------------
    # Classifier
    # ------------------------------------------------------------------
    op.create_table(
        "classifier_config",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("active_provider", sa.String(50)),
        sa.Column("openai_model", sa.String(200)),
        sa.Column("mistral_api_key", sa.String(500)),
        sa.Column("mistral_model", sa.String(200)),
        sa.Column("openrouter_api_key", sa.String(500)),
        sa.Column("openrouter_model", sa.String(200)),
        sa.Column("ollama_host", sa.String(500)),
        sa.Column("ollama_model", sa.String(200)),
        sa.Column("enable_title", sa.Boolean()),
        sa.Column("enable_tags", sa.Boolean()),
        sa.Column("enable_correspondent", sa.Boolean()),
        sa.Column("enable_document_type", sa.Boolean()),
        sa.Column("enable_storage_path", sa.Boolean()),
        sa.Column("enable_created_date", sa.Boolean()),
        sa.Column("enable_custom_fields", sa.Boolean()),
        sa.Column("tag_behavior", sa.String(50)),
        sa.Column("tags_min", sa.Integer()),
        sa.Column("tags_max", sa.Integer()),
        sa.Column("tags_keep_existing", sa.Boolean()),
        sa.Column("tags_ignore", sa.JSON()),
        sa.Column("tags_protected", sa.JSON()),
        sa.Column("dates_ignore", sa.JSON()),
        sa.Column("storage_path_behavior", sa.String(50)),
        sa.Column("storage_path_override_names", sa.JSON()),
        sa.Column("correspondent_behavior", sa.String(50)),
        sa.Column("correspondent_trim_prompt", sa.Boolean()),
        sa.Column("correspondent_strip_legal", sa.Boolean()),
        sa.Column("correspondent_ignore", sa.JSON()),
        sa.Column("review_mode", sa.String(50)),
        sa.Column("auto_classify_enabled", sa.Boolean()),
        sa.Column("auto_classify_interval", sa.Integer()),
        sa.Column("auto_classify_mode", sa.String(50)),
        sa.Column("auto_classify_skip_tag_ids", sa.JSON()),
        sa.Column("batch_size", sa.Integer()),
        sa.Column("prompt_title", sa.Text()),
        sa.Column("prompt_tags", sa.Text()),
        sa.Column("prompt_correspondent", sa.Text()),
        sa.Column("prompt_document_type", sa.Text()),
        sa.Column("prompt_date", sa.Text()),
        sa.Column("system_prompt", sa.Text()),
        sa.Column("excluded_tag_ids", sa.JSON()),
        sa.Column("excluded_correspondent_ids", sa.JSON()),
        sa.Column("excluded_document_type_ids", sa.JSON()),
        sa.Column("classification_tag_enabled", sa.Boolean()),
        sa.Column("classification_tag_name", sa.String(200)),
        sa.Column("review_tag_enabled", sa.Boolean()),
        sa.Column("review_tag_name", sa.String(200)),
        sa.Column("tag_ideas_tag_enabled", sa.Boolean()),
        sa.Column("tag_ideas_tag_name", sa.String(200)),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "classifier_storage_path_profiles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("paperless_path_id", sa.Integer(), nullable=False),
        sa.Column("paperless_path_name", sa.String(500)),
        sa.Column("paperless_path_path", sa.String(500)),
        sa.Column("enabled", sa.Boolean()),
        sa.Column("person_name", sa.String(300)),
        sa.Column("path_type", sa.String(50)),
        sa.Column("context_prompt", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("paperless_path_id"),
    )

    op.create_table(
        "classifier_custom_field_mappings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("paperless_field_id", sa.Integer(), nullable=False),
        sa.Column("paperless_field_name", sa.String(500)),
        sa.Column("paperless_field_type", sa.String(100)),
        sa.Column("enabled", sa.Boolean()),
        sa.Column("extraction_prompt", sa.Text()),
        sa.Column("example_values", sa.Text()),
        sa.Column("validation_regex", sa.String(500)),
        sa.Column("ignore_values", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("paperless_field_id"),
    )

    op.create_table(
        "classifier_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("document_title", sa.String(500)),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(200)),
        sa.Column("result_json", sa.JSON()),
        sa.Column("applied_json", sa.JSON()),
        sa.Column("tokens_input", sa.Integer()),
        sa.Column("tokens_output", sa.Integer()),
        sa.Column("cost_usd", sa.Float()),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("tool_calls_count", sa.Integer()),
        sa.Column("status", sa.String(50)),
        sa.Column("error_message", sa.Text()),
        sa.Column("tag_ideas", sa.JSON()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------
    op.create_table(
        "rag_config",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("embedding_provider", sa.String(100)),
        sa.Column("embedding_model", sa.String(200)),
        sa.Column("ollama_base_url", sa.String(500)),
        sa.Column("chunk_size", sa.Integer()),
        sa.Column("chunk_overlap", sa.Integer()),
        sa.Column("bm25_weight", sa.Float()),
        sa.Column("semantic_weight", sa.Float()),
        sa.Column("max_sources", sa.Integer()),
        sa.Column("max_context_tokens", sa.Integer()),
        sa.Column("chat_model_provider", sa.String(100)),
        sa.Column("chat_model", sa.String(200)),
        sa.Column("chat_system_prompt", sa.Text()),
        sa.Column("auto_index_enabled", sa.Boolean()),
        sa.Column("auto_index_interval", sa.Integer()),
        sa.Column("query_rewrite_enabled", sa.Boolean()),
        sa.Column("contextual_retrieval_enabled", sa.Boolean()),
        sa.Column("rag_enabled", sa.Boolean()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "rag_chat_sessions",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("title", sa.String(500)),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "rag_chat_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_rag_chat_messages_session_id", "rag_chat_messages", ["session_id"])

    op.create_table(
        "rag_indexing_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(50)),
        sa.Column("total_documents", sa.Integer()),
        sa.Column("indexed_documents", sa.Integer()),
        sa.Column("last_indexed_at", sa.DateTime()),
        sa.Column("error_message", sa.Text()),
        sa.Column("indexed_doc_ids", sa.Text()),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("key_hash", sa.String(128), nullable=False),
        sa.Column("key_prefix", sa.String(10), nullable=False),
        sa.Column("is_active", sa.Boolean()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("last_used_at", sa.DateTime()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash"),
    )

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    op.create_table(
        "ocr_page_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("total_pages", sa.Integer(), nullable=False),
        sa.Column("page_text", sa.Text()),
        sa.Column("status", sa.String(20)),
        sa.Column("error_message", sa.Text()),
        sa.Column("attempt_count", sa.Integer()),
        sa.Column("chars_extracted", sa.Integer()),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ocr_page_results_document_id", "ocr_page_results", ["document_id"])

    # ------------------------------------------------------------------
    # Cloud import
    # ------------------------------------------------------------------
    op.create_table(
        "cloud_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("source_type", sa.String(50)),
        sa.Column("enabled", sa.Boolean()),
        sa.Column("poll_interval_minutes", sa.Integer()),
        sa.Column("webdav_url", sa.String(500)),
        sa.Column("webdav_username", sa.String(200)),
        sa.Column("webdav_password", sa.Text()),
        sa.Column("webdav_path", sa.String(500)),
        sa.Column("rclone_remote", sa.String(100)),
        sa.Column("rclone_path", sa.String(500)),
        sa.Column("rclone_config", sa.Text()),
        sa.Column("local_path", sa.String(1000)),
        sa.Column("filename_prefix", sa.String(100)),
        sa.Column("paperless_tag_ids", sa.Text()),
        sa.Column("paperless_correspondent_id", sa.Integer()),
        sa.Column("paperless_document_type_id", sa.Integer()),
        sa.Column("after_import_action", sa.String(20)),
        sa.Column("last_checked_at", sa.DateTime()),
        sa.Column("last_status", sa.String(50)),
        sa.Column("last_error", sa.Text()),
        sa.Column("files_imported", sa.Integer()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "cloud_import_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("source_name", sa.String(200)),
        sa.Column("file_path", sa.String(1000), nullable=False),
        sa.Column("file_name", sa.String(500), nullable=False),
        sa.Column("paperless_doc_id", sa.Integer()),
        sa.Column("import_status", sa.String(20)),
        sa.Column("error_message", sa.Text()),
        sa.Column("imported_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_cloud_import_log_source_id", "cloud_import_log", ["source_id"])

    # ------------------------------------------------------------------
    # Duplicates
    # ------------------------------------------------------------------
    op.create_table(
        "duplicate_ignores",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("doc_id_a", sa.Integer(), nullable=False),
        sa.Column("doc_id_b", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "duplicate_invoice_cache",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("invoice_number", sa.String(200)),
        sa.Column("amount", sa.String(100)),
        sa.Column("extracted_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id"),
    )
    op.create_index(
        "ix_duplicate_invoice_cache_document_id", "duplicate_invoice_cache", ["document_id"]
    )

    # ------------------------------------------------------------------
    # Merge history
    # ------------------------------------------------------------------
    op.create_table(
        "merge_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("target_id", sa.Integer(), nullable=False),
        sa.Column("target_name", sa.String(500), nullable=False),
        sa.Column("merged_count", sa.Integer()),
        sa.Column("documents_affected", sa.Integer()),
        sa.Column("status", sa.String(50)),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "merge_history_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("merge_history_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("source_name", sa.String(500), nullable=False),
        sa.Column("document_ids", sa.JSON()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["merge_history_id"], ["merge_history.id"]),
    )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    op.create_table(
        "cleanup_statistics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("operation", sa.String(50), nullable=False),
        sa.Column("items_before", sa.Integer()),
        sa.Column("items_after", sa.Integer()),
        sa.Column("items_affected", sa.Integer()),
        sa.Column("documents_affected", sa.Integer()),
        sa.Column("details", sa.JSON()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "daily_stats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("date", sa.String(10), nullable=False),
        sa.Column("correspondents_total", sa.Integer()),
        sa.Column("tags_total", sa.Integer()),
        sa.Column("document_types_total", sa.Integer()),
        sa.Column("correspondents_merged", sa.Integer()),
        sa.Column("correspondents_deleted", sa.Integer()),
        sa.Column("tags_merged", sa.Integer()),
        sa.Column("tags_deleted", sa.Integer()),
        sa.Column("document_types_merged", sa.Integer()),
        sa.Column("document_types_deleted", sa.Integer()),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date"),
    )

    # ------------------------------------------------------------------
    # Cache / analysis
    # ------------------------------------------------------------------
    op.create_table(
        "paperless_cache",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cache_key", sa.String(100), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("count", sa.Integer()),
        sa.Column("updated_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cache_key"),
    )
    op.create_index("ix_paperless_cache_cache_key", "paperless_cache", ["cache_key"])

    op.create_table(
        "saved_analyses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("analysis_type", sa.String(50)),
        sa.Column("groups", sa.JSON(), nullable=False),
        sa.Column("stats", sa.JSON()),
        sa.Column("items_count", sa.Integer()),
        sa.Column("groups_count", sa.Integer()),
        sa.Column("processed_groups", sa.JSON()),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_table("saved_analyses")
    op.drop_index("ix_paperless_cache_cache_key", "paperless_cache")
    op.drop_table("paperless_cache")
    op.drop_table("daily_stats")
    op.drop_table("cleanup_statistics")
    op.drop_table("merge_history_items")
    op.drop_table("merge_history")
    op.drop_index("ix_duplicate_invoice_cache_document_id", "duplicate_invoice_cache")
    op.drop_table("duplicate_invoice_cache")
    op.drop_table("duplicate_ignores")
    op.drop_index("ix_cloud_import_log_source_id", "cloud_import_log")
    op.drop_table("cloud_import_log")
    op.drop_table("cloud_sources")
    op.drop_index("ix_ocr_page_results_document_id", "ocr_page_results")
    op.drop_table("ocr_page_results")
    op.drop_table("api_keys")
    op.drop_table("rag_indexing_state")
    op.drop_index("ix_rag_chat_messages_session_id", "rag_chat_messages")
    op.drop_table("rag_chat_messages")
    op.drop_table("rag_chat_sessions")
    op.drop_table("rag_config")
    op.drop_table("classifier_history")
    op.drop_table("classifier_custom_field_mappings")
    op.drop_table("classifier_storage_path_profiles")
    op.drop_table("classifier_config")
    op.drop_table("ignored_items")
    op.drop_table("ignored_tags")
    op.drop_table("custom_prompts")
    op.drop_table("app_settings")
    op.drop_table("llm_providers")
    op.drop_table("paperless_settings")
