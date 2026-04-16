from sqlalchemy import Column, Integer, String, Boolean, Text, DateTime
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
    
    id = Column(Integer, primary_key=True, default=1)
    url = Column(String(500), nullable=False, default="")
    api_token = Column(String(500), nullable=False, default="")
    is_configured = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class LLMProvider(Base):
    """LLM Provider configuration – connection config only (per D-05).

    Model/job routing is stored in AppSettings key-value (LLM_KEY_CLASSIFIER_MODEL etc.).
    A provider is implicitly configured if a row exists in this table.
    """
    __tablename__ = "llm_providers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)  # openai, anthropic, azure, ollama, mistral, openrouter
    display_name = Column(String(200), nullable=False)
    api_key = Column(String(500), default="")
    api_base_url = Column(String(500), default="")  # For Ollama or Azure
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    @property
    def is_ollama(self) -> bool:
        return self.name == "ollama"


class CustomPrompt(Base):
    """Custom prompts for different entity types."""
    __tablename__ = "custom_prompts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String(50), nullable=False)  # correspondents, tags, document_types
    prompt_template = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class IgnoredTag(Base):
    """Tags that should be ignored during cleanup analysis."""
    __tablename__ = "ignored_tags"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern = Column(String(500), nullable=False)  # Can be exact name or pattern
    reason = Column(String(500), default="")  # Why it's ignored
    is_regex = Column(Boolean, default=False)  # If true, treat as regex pattern
    created_at = Column(DateTime, server_default=func.now())


class IgnoredItem(Base):
    """Items (tags, correspondents, document_types) to ignore in specific analyses."""
    __tablename__ = "ignored_items"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, nullable=False)  # ID in Paperless
    item_name = Column(String(500), nullable=False)  # Name for display
    entity_type = Column(String(50), nullable=False)  # "tag", "correspondent", "document_type"
    analysis_type = Column(String(50), nullable=False)  # "nonsense", "correspondent_match", "doctype_match", "similar"
    reason = Column(String(500), default="")  # Optional: why it's ignored
    created_at = Column(DateTime, server_default=func.now())


class AppSettings(Base):
    """Application-wide settings.

    Uses split schema:
    - id=1, key=NULL → scalar UI settings (password_enabled, password_hash, etc.)
    - id=NULL, key=<string> → KV entries for LLM job routing (classifier_provider, etc.)

    The scalar row (id=1, key=NULL) holds UI settings. KV entries (id=NULL, key=<key>)
    store per-job provider/model routing (LLM-08).
    """
    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True)  # Scalar rows use id=1 explicitly; KV rows use id=None (DB autoincrement)
    # Key-value store for LLM job routing (LLM-08)
    key = Column(String(100), nullable=True, unique=True)  # NULL for scalar rows, string for KV entries
    value = Column(String(500), nullable=True)
    value_type = Column(String(20), default="str")  # str, int, bool, json
    # UI Password Protection
    password_enabled = Column(Boolean, default=False)
    password_hash = Column(String(500), default="")  # Hashed password
    # UI Options
    show_debug_menu = Column(Boolean, default=False)
    sidebar_compact = Column(Boolean, default=False)

    # NOTE: classifier_provider column is deprecated; use LLM_KEY_CLASSIFIER_PROVIDER KV entry
    classifier_provider = Column(String(100), default="ollama")

    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

__all__ = [
    "PaperlessSettings", "LLMProvider", "CustomPrompt", "IgnoredTag", "IgnoredItem",
    "AppSettings", "LLM_KEY_CLASSIFIER_PROVIDER", "LLM_KEY_CLASSIFIER_MODEL",
    "LLM_KEY_OCR_PROVIDER", "LLM_KEY_OCR_MODEL",
]

