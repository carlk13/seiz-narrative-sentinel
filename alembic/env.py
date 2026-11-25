"""Async Alembic migration environment configuration."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from src.models.db import Base, resolve_database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_effective_sqlalchemy_url() -> str:
    """Resolve effective database URL respecting explicit test overrides and env vars."""
    configured = config.get_main_option("sqlalchemy.url")
    env_url = os.getenv("DATABASE_URL") or os.getenv("SEIZ_DATABASE_URL")
    if configured and not configured.startswith("postgresql"):
        return resolve_database_url(configured)
    return resolve_database_url(env_url or configured)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = _get_effective_sqlalchemy_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute Alembic migration steps on a synchronous connection proxy."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async SQLAlchemy engine and execute migrations asynchronously."""
    section = dict(config.get_section(config.config_ini_section, {}))
    section["sqlalchemy.url"] = _get_effective_sqlalchemy_url()
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' async mode safely across event-loop contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(run_async_migrations())
        return

    with ThreadPoolExecutor(max_workers=1) as pool_exec:
        pool_exec.submit(asyncio.run, run_async_migrations()).result()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
