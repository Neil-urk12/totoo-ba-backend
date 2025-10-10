# app/core/database.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()

print("Initializing database...")

try:
    DATABASE_URL = os.getenv("DATABASE_URL")
    print(f"Using database URL: {DATABASE_URL}")

    engine = create_engine(DATABASE_URL)


    with engine.connect() as connection:
        print("Connection successful")

except Exception as e:
    print(f"Failed to connect: {e}")


# engine = create_engine(
#     DATABASE_URL,
#     echo=True,
#     poolclass=NullPool,  # Queue Pool ligma
# )

# SessionLocal = sessionmaker(
#     autocommit=False,
#     autoflush=False,
#     bind=engine
# )

# Base = declarative_base()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# def test_connection():
#     """
#     Test the database connection by executing a simple query.
#     """
#     try:
#         with engine.connect() as conn:
#             # Execute a simple query to test the connection
#             result = conn.execute(text("SELECT 1"))
#         print("Database connection successful.")
#         return True
#     except Exception as e:
#         print(f"Database connection failed: {e}")
#         return False

print("Database initialization complete.")

# if __name__ == "__main__":
#     test_connection()
