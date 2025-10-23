# app/core/database.py
"""Database configuration and session management.

This module provides both synchronous and asynchronous database engines,
session makers, and connection testing utilities for PostgreSQL using SQLAlchemy.
"""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

load_dotenv()


try:
    DATABASE_URL = os.getenv("DATABASE_URL")
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

    # Synchronous engine for sync operations
    sync_engine = create_engine(DATABASE_URL)

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

    # Remove the synchronous connection attempt since it causes issues in async contexts

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
    """Get a synchronous database session.
    
    This is a generator function that yields a database session and ensures
    proper cleanup after use. Intended for use with FastAPI's Depends.
    
    Yields:
        Session: A SQLAlchemy synchronous database session.
        
    Raises:
        RuntimeError: If the database connection is not available.
        
    Example:
        ```python
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
        ```
    """
    if SessionLocal is None:
        raise RuntimeError("Database connection not available")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




def test_connection():
    """Test the synchronous database connection.
    
    Executes a simple query on the food_products table to verify that
    the database connection is working properly.
    
    Returns:
        bool: True if the connection is successful, False otherwise.
        
    Example:
        ```python
        if test_connection():
            print("Database is connected")
        ```
    """
    try:
        with sync_engine.connect() as conn:
            # Execute a query to test the connection and fetch 10 items from food_products
            result = conn.execute(text("SELECT * FROM food_products LIMIT 10"))
            result.fetchall()
        return True
    except Exception:
        return False


async def test_connection_async():
    """Test the asynchronous database connection.
    
    Async version of test_connection for use in async contexts.
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



# if __name__ == "__main__":
#     test_connection()
