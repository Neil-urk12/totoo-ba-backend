# app/core/database.py
from sqlalchemy import create_engine, func, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings


# ----------------------------------------------------------------------------
# Engine configuration using application Settings
# ----------------------------------------------------------------------------
try:
    settings = get_settings()

    # Synchronous engine (for migrations/occasional sync tasks)
    sync_engine = create_engine(
        settings.database_url_sync,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout,
        pool_recycle=settings.database_pool_recycle,
        future=True,
    )

    # Async engine for primary application operations
    async_engine = create_async_engine(
        settings.database_url,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout,
        pool_recycle=settings.database_pool_recycle,
        future=True,
    )

except Exception:
    # Fallback to None if configuration fails (app will log at startup)
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




def test_connection() -> bool:
    """Lightweight synchronous DB health check (SELECT 1)."""
    try:
        if sync_engine is None:
            return False
        with sync_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def test_connection_async() -> bool:
    """Lightweight async DB health check (SELECT 1)."""
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
