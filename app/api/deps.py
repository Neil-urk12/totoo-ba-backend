"""
Dependency injection for FastAPI endpoints.
Provides database sessions and repository instances.
"""

from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import async_session
from app.api.repository.products_repository import ProductsRepository
from app.services.product_verification_service import ProductVerificationService


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an async database session with automatic transaction management.

    This provides per-request transactions that:
    - Automatically commit on successful completion
    - Automatically rollback on any exceptions
    - Allow repositories to use flush() for immediate DB sync without committing

    Yields:
        AsyncSession: Database session for async operations
    """
    if async_session is None:
        raise RuntimeError("Database connection not available - async_session is None")

    async with async_session() as session:
        try:
            yield session
            await session.commit()  # Auto-commit on success
        except Exception:
            await session.rollback()  # Auto-rollback on error
            raise
        finally:
            await session.close()


async def get_transactional_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a session WITHOUT automatic commit.
    Use this when you need manual transaction control in service layer.

    Yields:
        AsyncSession: Database session for manual transaction management
    """
    if async_session is None:
        raise RuntimeError("Database connection not available - async_session is None")

    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


def get_products_repository(
    session: AsyncSession = Depends(get_async_session),
) -> ProductsRepository:
    """
    Dependency that provides a ProductsRepository with automatic transaction management.

    Args:
        session: Database session with auto-commit/rollback

    Returns:
        ProductsRepository: Repository instance with session
    """
    return ProductsRepository(session=session)


def get_products_repository_transactional(
    session: AsyncSession = Depends(get_transactional_session),
) -> ProductsRepository:
    """
    Dependency that provides a ProductsRepository with manual transaction control.
    Use this when you need to handle commits/rollbacks manually in service layer.

    Args:
        session: Database session for manual transaction management

    Returns:
        ProductsRepository: Repository instance with session
    """
    return ProductsRepository(session=session)


def get_product_verification_service(
    products_repo: ProductsRepository = Depends(get_products_repository),
) -> ProductVerificationService:
    """
    Dependency that provides a ProductVerificationService with business logic.

    Args:
        products_repo: Repository for data access

    Returns:
        ProductVerificationService: Service instance with business logic
    """
    return ProductVerificationService(products_repo=products_repo)
