# app/core/database.py
"""Database configuration and session management.

Provides the asynchronous PostgreSQL engine, sessions, and connection probe.
"""
import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

load_dotenv()


try:
    DATABASE_URL = os.getenv("DATABASE_URL")
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

    # Async engine for async operations with better timeout handling
    async_engine = create_async_engine(
        ASYNC_DATABASE_URL,
        poolclass=NullPool,
        echo=False,
        connect_args={
            "timeout": 60,  # Connection timeout in seconds
            "command_timeout": 60,  # Command timeout in seconds
        },
    )

except Exception:
    async_engine = None


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


async def test_connection_async():
    """Test the asynchronous database connection.

    Tests both raw SQL queries and ORM queries to ensure the async
    database engine is functioning correctly.

    Returns:
        bool: True if the connection is successful, False otherwise.

    Example:
        ```python
        if await test_connection_async():
            print("Async database is connected")
        ```
    """
    if async_engine is None or async_session is None:
        return False

    try:
        # Test raw SQL query
        async with async_engine.begin() as conn:
            result = await conn.execute(text("SELECT * FROM food_products LIMIT 10"))
            result.fetchall()

        # Test ORM query
        from sqlalchemy import select

        from app.models.food_products import FoodProducts

        async with async_session() as session:
            query = select(FoodProducts).limit(10)
            result = await session.execute(query)
            result.scalars().all()

        return True
    except Exception:
        return False
