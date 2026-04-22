from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.database import Base


class AuthConfig(Base):
    __tablename__ = "auth_config"
    id = Column(Integer, primary_key=True, default=1)
    password_hash = Column(String(500), default="")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
