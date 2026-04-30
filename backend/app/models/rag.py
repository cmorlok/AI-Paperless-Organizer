from datetime import datetime
import uuid
from sqlalchemy import Integer, String, Boolean, Text, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


def _generate_uuid():
    return str(uuid.uuid4())


class RagChatSession(Base):
    """A chat session / conversation."""
    __tablename__ = "rag_chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    title: Mapped[str] = mapped_column(String(500), default="Neuer Chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class RagChatMessage(Base):
    """A single message in a chat session."""
    __tablename__ = "rag_chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of source references
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RagIndexingState(Base):
    """Tracks the current state of RAG document indexing."""
    __tablename__ = "rag_indexing_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    status: Mapped[str] = mapped_column(String(50), default="idle")  # idle, indexing, completed, error
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    indexed_documents: Mapped[int] = mapped_column(Integer, default=0)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, default="")
    indexed_doc_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of indexed Paperless doc IDs
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class RagConfig(Base):
    """RAG system configuration (non-provider fields only; provider/model migrated to KV)."""
    __tablename__ = "rag_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    chunk_size: Mapped[int] = mapped_column(Integer, default=500)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=50)
    bm25_weight: Mapped[float] = mapped_column(Float, default=0.3)
    semantic_weight: Mapped[float] = mapped_column(Float, default=0.7)
    max_sources: Mapped[int] = mapped_column(Integer, default=8)
    max_context_tokens: Mapped[int] = mapped_column(Integer, default=4000)
    chat_system_prompt: Mapped[str] = mapped_column(Text, default="Du bist ein hilfreicher Assistent, der Fragen zu Dokumenten beantwortet. Antworte basierend auf dem bereitgestellten Kontext. Wenn du die Antwort nicht im Kontext findest, sage das ehrlich.")
    auto_index_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_index_interval: Mapped[int] = mapped_column(Integer, default=30)
    query_rewrite_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    contextual_retrieval_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    rag_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class ApiKey(Base):
    """API keys for external access to RAG endpoints."""
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(String(10), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
