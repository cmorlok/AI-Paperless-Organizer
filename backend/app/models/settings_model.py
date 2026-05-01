from datetime import datetime
from sqlalchemy import Integer, String, Boolean, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base

# LLM Job Routing Keys (D-04, LLM-08)
LLM_KEY_CLASSIFIER_PROVIDER = "classifier_provider"
LLM_KEY_CLASSIFIER_MODEL = "classifier_model"
LLM_KEY_OCR_PROVIDER = "ocr_provider"
LLM_KEY_OCR_MODEL = "ocr_model"


class PaperlessSettings(Base):
    """Paperless-ngx connection settings."""
    __tablename__ = "paperless_settings"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    url: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    api_token: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    is_configured: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class LLMProvider(Base):
    """LLM Provider configuration – connection config only (per D-05).

    Model/job routing is stored in AppSettings key-value (LLM_KEY_CLASSIFIER_MODEL etc.).
    A provider is implicitly configured if a row exists in this table.
    """
    __tablename__ = "llm_providers"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)  # openai, anthropic, azure, ollama, mistral, openrouter
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    api_key: Mapped[str] = mapped_column(String(500), default="")
    api_base_url: Mapped[str] = mapped_column(String(500), default="")  # For Ollama or Azure
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class CustomPrompt(Base):
    """Custom prompts for different entity types."""
    __tablename__ = "custom_prompts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # correspondents, tags, document_types
    prompt_template: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    modified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class IgnoredTag(Base):
    """Tags that should be ignored during cleanup analysis."""
    __tablename__ = "ignored_tags"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern: Mapped[str] = mapped_column(String(500), nullable=False)  # Can be exact name or pattern
    reason: Mapped[str] = mapped_column(String(500), default="")  # Why it's ignored
    is_regex: Mapped[bool] = mapped_column(Boolean, default=False)  # If true, treat as regex pattern
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class IgnoredItem(Base):
    """Items (tags, correspondents, document_types) to ignore in specific analyses."""
    __tablename__ = "ignored_items"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(Integer, nullable=False)  # ID in Paperless
    item_name: Mapped[str] = mapped_column(String(500), nullable=False)  # Name for display
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "tag", "correspondent", "document_type"
    analysis_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "nonsense", "correspondent_match", "doctype_match", "similar"
    reason: Mapped[str] = mapped_column(String(500), default="")  # Optional: why it's ignored
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AppSettings(Base):
    """Application-wide settings.

    Uses split schema:
    - id=1, key=NULL → scalar UI settings (show_debug_menu, sidebar_compact, etc.)
    - id=NULL, key=<string> → KV entries for LLM job routing (classifier_provider, etc.)

    The scalar row (id=1, key=NULL) holds UI settings. KV entries (id=NULL, key=<key>)
    store per-job provider/model routing (LLM-08).

    Auth settings (password_hash) have moved to the AuthConfig model.
    """
    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # Scalar rows use id=1 explicitly; KV rows use id=None (DB autoincrement)
    # Key-value store for LLM job routing (LLM-08)
    key: Mapped[str | None] = mapped_column(String(100), nullable=True, unique=True)  # NULL for scalar rows, string for KV entries
    value: Mapped[str | None] = mapped_column(String(500), nullable=True)
    value_type: Mapped[str] = mapped_column(String(20), default="str")  # str, int, bool, json
    # UI Options
    show_debug_menu: Mapped[bool] = mapped_column(Boolean, default=False)
    sidebar_compact: Mapped[bool] = mapped_column(Boolean, default=False)

    # NOTE: classifier_provider column is deprecated; use LLM_KEY_CLASSIFIER_PROVIDER KV entry
    classifier_provider: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

__all__ = [
    "PaperlessSettings", "LLMProvider", "CustomPrompt", "IgnoredTag", "IgnoredItem",
    "AppSettings", "LLM_KEY_CLASSIFIER_PROVIDER", "LLM_KEY_CLASSIFIER_MODEL",
    "LLM_KEY_OCR_PROVIDER", "LLM_KEY_OCR_MODEL",
]

