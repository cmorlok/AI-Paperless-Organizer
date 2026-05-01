"""Models for the KI-Klassifizierer feature."""

from datetime import datetime
from typing import Any

from sqlalchemy import Integer, String, Boolean, Text, DateTime, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


class ClassifierConfig(Base):
    """Main configuration for the document classifier."""
    __tablename__ = "classifier_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)

    # Active provider: "openai", "mistral", "openrouter", or "ollama"
    active_provider: Mapped[str] = mapped_column(String(50), default="openai")

    # OpenAI settings (reuses existing LLMProvider for api_key)
    openai_model: Mapped[str] = mapped_column(String(200), default="gpt-4o-mini")

    # Mistral settings
    mistral_api_key: Mapped[str] = mapped_column(String(500), default="")
    mistral_model: Mapped[str] = mapped_column(String(200), default="mistral-small-latest")

    # OpenRouter settings (openai-compatible, many models)
    openrouter_api_key: Mapped[str] = mapped_column(String(500), default="")
    openrouter_model: Mapped[str] = mapped_column(String(200), default="mistral/mistral-small-3.1-24b-instruct")

    # Which fields to classify
    enable_title: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_tags: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_correspondent: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_document_type: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_storage_path: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_created_date: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_custom_fields: Mapped[bool] = mapped_column(Boolean, default=False)

    # Tag behavior: "existing_only", "suggest_new", "auto_create"
    tag_behavior: Mapped[str] = mapped_column(String(50), default="existing_only")
    tags_min: Mapped[int] = mapped_column(Integer, default=1)
    tags_max: Mapped[int] = mapped_column(Integer, default=5)
    tags_keep_existing: Mapped[bool] = mapped_column(Boolean, default=True)
    # JSON array of tag names/patterns to never suggest
    tags_ignore: Mapped[Any] = mapped_column(JSON, default=[])
    # JSON array of tag names/patterns to keep when replacing (e.g. INBOX, ocr*)
    tags_protected: Mapped[Any] = mapped_column(JSON, default=[])
    # JSON array of date strings to never use as created_date (e.g. birthdays "1987-06-17")
    dates_ignore: Mapped[Any] = mapped_column(JSON, default=[])
    # Storage path assignment behavior:
    #   "always"          -- always apply AI suggestion
    #   "keep_if_set"     -- never change if document already has a path
    #   "keep_except_list"-- keep existing UNLESS current path is in override_names list
    storage_path_behavior: Mapped[str] = mapped_column(String(50), default="always")
    # JSON array of path names that should be overridden even in keep_except_list mode
    storage_path_override_names: Mapped[Any] = mapped_column(JSON, default=["Zuweisen"])
    # Correspondent behavior: "existing_only", "suggest_new"
    correspondent_behavior: Mapped[str] = mapped_column(String(50), default="existing_only")
    # Correspondent name trimming options
    correspondent_trim_prompt: Mapped[bool] = mapped_column(Boolean, default=False)
    correspondent_strip_legal: Mapped[bool] = mapped_column(Boolean, default=False)
    # JSON array of names to never suggest as correspondent (e.g. person names used as storage paths)
    correspondent_ignore: Mapped[Any] = mapped_column(JSON, default=[])

    # Review mode: "always", "uncertain_only", "auto_apply"
    review_mode: Mapped[str] = mapped_column(String(50), default="always")

    # Auto-classification background job
    auto_classify_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_classify_interval: Mapped[int] = mapped_column(Integer, default=5)  # minutes
    # What to do after auto-classify: "review" (always review), "auto_apply" (apply if confident)
    auto_classify_mode: Mapped[str] = mapped_column(String(50), default="review")
    # Tags that mark a document to be skipped entirely by auto-classification
    auto_classify_skip_tag_ids: Mapped[Any] = mapped_column(JSON, default=[])

    # Batch settings
    batch_size: Mapped[int] = mapped_column(Integer, default=10)

    # Per-field prompt hints (user customization)
    prompt_title: Mapped[str] = mapped_column(Text, default="")
    prompt_tags: Mapped[str] = mapped_column(Text, default="")
    prompt_correspondent: Mapped[str] = mapped_column(Text, default="")
    prompt_document_type: Mapped[str] = mapped_column(Text, default="")
    prompt_date: Mapped[str] = mapped_column(Text, default="")

    # Excluded items: JSON arrays of Paperless IDs to skip
    excluded_tag_ids: Mapped[Any] = mapped_column(JSON, default=[])
    excluded_correspondent_ids: Mapped[Any] = mapped_column(JSON, default=[])
    excluded_document_type_ids: Mapped[Any] = mapped_column(JSON, default=[])

    # Classification Tag: optional tag assigned to every classified document
    classification_tag_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    classification_tag_name: Mapped[str] = mapped_column(String(200), default="KI-klassifiziert")
    # Review Tag: optional tag assigned when document goes into review queue
    review_tag_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    review_tag_name: Mapped[str] = mapped_column(String(200), default="KI-prüfen")
    # Tag-Ideas Tag: optional tag assigned when AI suggests new tags (tag ideas)
    tag_ideas_tag_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    tag_ideas_tag_name: Mapped[str] = mapped_column(String(200), default="KI-tag-ideen")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class StoragePathProfile(Base):
    """Person profile linked to a Paperless storage path."""
    __tablename__ = "classifier_storage_path_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paperless_path_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    paperless_path_name: Mapped[str] = mapped_column(String(500), default="")
    paperless_path_path: Mapped[str] = mapped_column(String(500), default="")

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    person_name: Mapped[str] = mapped_column(String(300), default="")
    # "private" or "business"
    path_type: Mapped[str] = mapped_column(String(50), default="private")
    context_prompt: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class CustomFieldMapping(Base):
    """Configuration for a single Paperless custom field extraction."""
    __tablename__ = "classifier_custom_field_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paperless_field_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    paperless_field_name: Mapped[str] = mapped_column(String(500), default="")
    paperless_field_type: Mapped[str] = mapped_column(String(100), default="string")

    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    extraction_prompt: Mapped[str] = mapped_column(Text, default="")
    example_values: Mapped[str] = mapped_column(Text, default="")
    validation_regex: Mapped[str] = mapped_column(String(500), default="")
    ignore_values: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class ClassificationHistory(Base):
    """Log of document classifications performed."""
    __tablename__ = "classifier_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(Integer, nullable=False)
    document_title: Mapped[str] = mapped_column(String(500), default="")

    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(200), default="")

    # What was classified
    result_json: Mapped[Any] = mapped_column(JSON, default={})
    # What was applied (may differ after review)
    applied_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)

    # Metrics
    tokens_input: Mapped[int] = mapped_column(Integer, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    tool_calls_count: Mapped[int] = mapped_column(Integer, default=0)

    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending, applied, rejected, error
    error_message: Mapped[str] = mapped_column(Text, default="")

    # New tag ideas suggested by AI but not yet created
    tag_ideas: Mapped[Any] = mapped_column(JSON, default=[])

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
