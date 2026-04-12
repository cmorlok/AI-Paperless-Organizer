from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
)

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


def run_migrations() -> None:
    """Run Alembic migrations to head.

    For databases that pre-date Alembic (i.e. they have tables but no
    alembic_version row), the database is automatically stamped at head so
    that the already-applied schema changes are not re-executed.
    """
    from sqlalchemy import create_engine, inspect
    from alembic.config import Config
    from alembic import command

    ini_path = Path(__file__).parent.parent / "alembic.ini"
    cfg = Config(str(ini_path))

    sync_url = settings.database_url.replace("sqlite+aiosqlite", "sqlite")
    sync_engine = create_engine(sync_url)

    inspector = inspect(sync_engine)
    table_names = inspector.get_table_names()
    sync_engine.dispose()

    is_pre_alembic = "classifier_config" in table_names and "alembic_version" not in table_names

    if is_pre_alembic:
        # Existing installation: all DDL was already applied by the old
        # _migrate_columns system — just record that we're at head.
        command.stamp(cfg, "head")
    else:
        command.upgrade(cfg, "head")


async def get_db():
    """Dependency to get database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
