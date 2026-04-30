"""Model for saved AI analysis results."""

from datetime import datetime
from typing import Any

from sqlalchemy import Integer, String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


class SavedAnalysis(Base):
    """Store AI analysis results for later use."""
    __tablename__ = "saved_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'correspondents', 'tags', 'document_types'
    analysis_type: Mapped[str] = mapped_column(String(50), default="similarity")  # 'similarity', 'nonsense', etc.
    groups: Mapped[Any] = mapped_column(JSON, nullable=False)  # The actual analysis results
    stats: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # Token usage, etc.
    items_count: Mapped[int] = mapped_column(Integer, default=0)  # How many items were analyzed
    groups_count: Mapped[int] = mapped_column(Integer, default=0)  # How many groups were found
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Track which groups have been processed
    processed_groups: Mapped[Any] = mapped_column(JSON, default=list)  # List of processed group indices

