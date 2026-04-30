from datetime import datetime
from sqlalchemy import Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


class DuplicateIgnore(Base):
    """A pair of document IDs that the user marked as 'not a duplicate'."""
    __tablename__ = "duplicate_ignores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id_a: Mapped[int] = mapped_column(Integer, nullable=False)
    doc_id_b: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DuplicateInvoiceCache(Base):
    """Cached invoice extraction results (number + amount) per document."""
    __tablename__ = "duplicate_invoice_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    invoice_number: Mapped[str] = mapped_column(String(200), default="")
    amount: Mapped[str] = mapped_column(String(100), default="")
    extracted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
