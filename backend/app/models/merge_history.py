from datetime import datetime
from typing import Any

from sqlalchemy import Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class MergeHistory(Base):
    """History of merge operations for potential rollback."""
    __tablename__ = "merge_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # correspondents, tags, document_types
    target_id: Mapped[int] = mapped_column(Integer, nullable=False)
    target_name: Mapped[str] = mapped_column(String(500), nullable=False)
    merged_count: Mapped[int] = mapped_column(Integer, default=0)
    documents_affected: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default="completed")  # completed, rolled_back
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    items = relationship("MergeHistoryItem", back_populates="merge_history", cascade="all, delete-orphan")


class MergeHistoryItem(Base):
    """Individual items that were merged."""
    __tablename__ = "merge_history_items"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    merge_history_id: Mapped[int] = mapped_column(Integer, ForeignKey("merge_history.id"), nullable=False)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_name: Mapped[str] = mapped_column(String(500), nullable=False)
    document_ids: Mapped[Any] = mapped_column(JSON, default=list)  # List of document IDs that were updated
    
    merge_history = relationship("MergeHistory", back_populates="items")

