from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings
import os

os.makedirs("data", exist_ok=True)

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


def run_migrations() -> None:
    """Run Alembic migrations to head.

    Pre-Alembic databases (tables exist, no alembic_version) were created by
    the old _migrate_columns system. Their schema matches 0001, so we stamp at
    0001 and run subsequent migrations. Fresh databases start at base.
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
    has_alembic = "alembic_version" in table_names

    if has_alembic:
        command.upgrade(cfg, "head")
        sync_engine.dispose()
        return

    # Pre-Alembic DB: has tables but no alembic_version
    # Schema matches 0001 (old _migrate_columns created all tables)
    # Stamp at 0001 so 0001's upgrade is skipped, run 0002 onwards
    if "classifier_config" in table_names:
        command.stamp(cfg, "0001")
        sync_engine.dispose()
        command.upgrade(cfg, "head")
        return

    # Fresh DB: no tables, run full migration from base
    sync_engine.dispose()
    command.upgrade(cfg, "head")


async def get_db():
    """Dependency to get database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
