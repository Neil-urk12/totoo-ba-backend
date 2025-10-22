# app/core/database.py
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Centralized settings
from app.core.config import get_settings


# Engines are configured from Settings to avoid env duplication and enable tuning
try:
    settings = get_settings()

    # Synchronous engine (used by data loaders and optional health checks)
    # Keep small pool to avoid exhausting DB connections during API traffic
    sync_engine = create_engine(
        settings.database_url_sync,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_size=max(1, min(settings.database_pool_size, 5)),
        max_overflow=0,
        pool_timeout=settings.database_pool_timeout,
        pool_recycle=settings.database_pool_recycle,
    )

    # Async engine for the API path
    async_engine = create_async_engine(
        settings.database_url,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout,
        pool_recycle=settings.database_pool_recycle,
        connect_args={
            # asyncpg specific timeouts
            "timeout": 60,
            "command_timeout": 60,
        },
    )
except Exception:
    sync_engine = None
    async_engine = None


# Sync sessionmaker
if sync_engine:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
else:
    SessionLocal = None

# Async sessionmaker
if async_engine:
    async_session = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
else:
    async_session = None

# Base for declarative models
Base = declarative_base()

# Export the async engine with the same name for backward compatibility with async code
engine = async_engine


def get_db():
    if SessionLocal is None:
        raise RuntimeError("Database connection not available")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




def test_connection():
    """Fast, lightweight connectivity check (no heavy table scans)."""
    try:
        if sync_engine is None:
            return False
        with sync_engine.connect() as conn:
            _ = conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def test_connection_async():
    """Fast async connectivity check using a trivial statement."""
    if async_engine is None:
        return False

    try:
        async with async_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False



# if __name__ == "__main__":
#     test_connection()
