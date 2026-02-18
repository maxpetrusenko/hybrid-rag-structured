"""Database connection and utilities."""

import os
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from .models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://hybridrag:hybridrag@localhost:5432/hybridrag")

# Convert postgresql:// to postgresql+asyncpg:// for async
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Async engine
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

async_session_maker = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Sync engine for migrations
sync_engine_url = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")
sync_engine = create_engine(sync_engine_url)


class Database:
    """Database connection manager."""

    @staticmethod
    async def init_db():
        """Initialize database schema."""
        async with async_engine.begin() as conn:
            # Enable pgvector
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # Create tables
            await conn.run_sync(Base.metadata.create_all)

    @staticmethod
    async def close():
        """Close database connections."""
        await async_engine.dispose()

    @staticmethod
    async def get_session() -> AsyncGenerator[AsyncSession, None]:
        """Get async session."""
        async with async_session_maker() as session:
            yield session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting DB session."""
    async with async_session_maker() as session:
        yield session
