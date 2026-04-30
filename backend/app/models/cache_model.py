"""Persistent cache model for storing Paperless data."""

from datetime import datetime
from typing import Any

from sqlalchemy import Integer, String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


class PaperlessCache(Base):
    """Persistent cache for Paperless-ngx data."""
    __tablename__ = "paperless_cache"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cache_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    data: Mapped[Any] = mapped_column(JSON, nullable=False)
    count: Mapped[int] = mapped_column(Integer, default=0)  # Quick access to count without parsing JSON
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

