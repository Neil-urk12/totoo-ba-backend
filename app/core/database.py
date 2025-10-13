# app/core/database.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()

print("Initializing database...")

try:
    DATABASE_URL = os.getenv("DATABASE_URL")
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    print(f"Using database URL: {DATABASE_URL}")

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

    with sync_engine.connect() as connection:
        print("Connection successful")

except Exception as e:
    print(f"Failed to connect: {e}")


# Sync sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# Async sessionmaker
async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Base for declarative models
Base = declarative_base()

# Export the async engine with the same name for backward compatibility with async code
engine = async_engine


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from app.models.food_products import FoodProducts


def test_connection():
    """
    Test the database connection by executing a query on food_products table.
    """
    try:
        with sync_engine.connect() as conn:
            # Execute a query to test the connection and fetch 10 items from food_products
            result = conn.execute(text("SELECT * FROM food_products LIMIT 10"))
            rows = result.fetchall()
            print(
                f"Database connection successful. Found {len(rows)} records in food_products table."
            )
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


async def test_connection_async():
    """
    Async version of test_connection for use in async contexts.
    Tests both raw SQL and ORM queries.
    """
    try:
        # Test raw SQL query
        async with async_engine.begin() as conn:
            result = await conn.execute(text("SELECT * FROM food_products LIMIT 10"))
            rows = result.fetchall()
            print(
                f"✅ Raw SQL query successful. Found {len(rows)} records in food_products table."
            )

        # Test ORM query
        from sqlalchemy import select
        from app.models.food_products import FoodProducts

        async with async_session() as session:
            query = (
                select(FoodProducts)
                .limit(10)
            )
            result = await session.execute(query)
            products = result.scalars().all()
            print(
                f"✅ ORM query successful. Found {len(products)} food products using SQLAlchemy ORM."
            )

        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


print("Database initialization complete.")

# if __name__ == "__main__":
#     test_connection()
