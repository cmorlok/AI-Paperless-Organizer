import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure the backend root is on sys.path so `app.*` imports work both when
# Alembic is invoked from the CLI and when called programmatically at runtime.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import Base first, then all models so their tables are registered on
# Base.metadata before autogenerate or create_all inspects it.
from app.database import Base  # noqa: E402
import app.models  # noqa: E402, F401

# Override the URL from the application config so that DATABASE_URL env vars
# and any custom paths are respected.  Alembic uses the synchronous SQLite
# driver; we replace the async aiosqlite driver here.
from app.config import settings  # noqa: E402

config = context.config
config.set_main_option(
    "sqlalchemy.url",
    settings.database_url.replace("sqlite+aiosqlite", "sqlite"),
)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live database connection (emit SQL to stdout)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
