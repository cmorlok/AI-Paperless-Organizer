"""Statistics tracking for cleanup operations."""

from datetime import datetime
from typing import Any

from sqlalchemy import Integer, String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


class CleanupStatistics(Base):
    """Track cleanup operations and savings."""
    __tablename__ = "cleanup_statistics"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # correspondents, tags, document_types
    operation: Mapped[str] = mapped_column(String(50), nullable=False)  # merge, delete, cleanup
    items_before: Mapped[int] = mapped_column(Integer, default=0)
    items_after: Mapped[int] = mapped_column(Integer, default=0)
    items_affected: Mapped[int] = mapped_column(Integer, default=0)  # How many items were merged/deleted
    documents_affected: Mapped[int] = mapped_column(Integer, default=0)  # How many documents were updated
    details: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # Additional details like names
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DailyStats(Base):
    """Daily aggregated statistics."""
    __tablename__ = "daily_stats"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)  # YYYY-MM-DD
    
    # Counts at end of day
    correspondents_total: Mapped[int] = mapped_column(Integer, default=0)
    tags_total: Mapped[int] = mapped_column(Integer, default=0)
    document_types_total: Mapped[int] = mapped_column(Integer, default=0)
    
    # Operations this day
    correspondents_merged: Mapped[int] = mapped_column(Integer, default=0)
    correspondents_deleted: Mapped[int] = mapped_column(Integer, default=0)
    tags_merged: Mapped[int] = mapped_column(Integer, default=0)
    tags_deleted: Mapped[int] = mapped_column(Integer, default=0)
    document_types_merged: Mapped[int] = mapped_column(Integer, default=0)
    document_types_deleted: Mapped[int] = mapped_column(Integer, default=0)
    
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

