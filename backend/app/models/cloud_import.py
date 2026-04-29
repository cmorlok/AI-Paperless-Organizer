from datetime import datetime
from sqlalchemy import Integer, String, Boolean, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.database import Base


class CloudSource(Base):
    """A cloud storage source to monitor and import from."""
    __tablename__ = "cloud_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="Neue Quelle")
    source_type: Mapped[str] = mapped_column(String(50), default="webdav")  # webdav, rclone, local
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    poll_interval_minutes: Mapped[int] = mapped_column(Integer, default=5)

    # WebDAV connection
    webdav_url: Mapped[str] = mapped_column(String(500), default="")
    webdav_username: Mapped[str] = mapped_column(String(200), default="")
    webdav_password: Mapped[str] = mapped_column(Text, default="")
    webdav_path: Mapped[str] = mapped_column(String(500), default="/")

    # rclone connection
    rclone_remote: Mapped[str] = mapped_column(String(100), default="")
    rclone_path: Mapped[str] = mapped_column(String(500), default="/")
    rclone_config: Mapped[str] = mapped_column(Text, default="")  # content of rclone.conf for this remote

    # Local folder
    local_path: Mapped[str] = mapped_column(String(1000), default="")

    # Import settings for Paperless
    filename_prefix: Mapped[str] = mapped_column(String(100), default="")
    paperless_tag_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of tag IDs
    paperless_correspondent_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paperless_document_type_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    after_import_action: Mapped[str] = mapped_column(String(20), default="keep")  # keep, delete

    # Status
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[str] = mapped_column(String(50), default="idle")  # idle, syncing, error
    last_error: Mapped[str] = mapped_column(Text, default="")
    files_imported: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class CloudImportLog(Base):
    """Log of imported files per source."""
    __tablename__ = "cloud_import_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    source_name: Mapped[str] = mapped_column(String(200), default="")
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    paperless_doc_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    import_status: Mapped[str] = mapped_column(String(20), default="success")  # success, error, skipped
    error_message: Mapped[str] = mapped_column(Text, default="")
    imported_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
