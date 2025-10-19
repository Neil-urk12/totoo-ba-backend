"""
Base repository class for database operations.
Provides common database operations and patterns for all repositories.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from sqlalchemy import and_, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

# Type variable for model classes
ModelType = TypeVar("ModelType", bound=declarative_base())


class BaseRepository(Generic[ModelType], ABC):
    """
    Base repository class providing common database operations.
    All specific repositories should inherit from this class.
    """

    def __init__(self, model: type[ModelType], session: AsyncSession):
        """
        Initialize repository with the model class and session.

        Args:
            model: SQLAlchemy model class
            session: AsyncSession for database operations
        """
        self.model = model
        self.session = session

    async def get_by_id(self, id_value: Any) -> ModelType | None:
        """
        Get a single record by ID.

        Args:
            id_value: The ID value to search for

        Returns:
            Model instance or None if not found
        """
        query = select(self.model).where(self.model.id == id_value)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[ModelType]:
        """
        Get all records with optional pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of model instances
        """
        query = select(self.model)
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def create(self, **kwargs) -> ModelType:
        """
        Create a new record.

        Args:
            **kwargs: Field values for the new record

        Returns:
            Created model instance
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()  # Send to DB but don't commit
        await self.session.refresh(instance)
        return instance

    async def update(self, id_value: Any, **kwargs) -> ModelType | None:
        """
        Update a record by ID.

        Args:
            id_value: The ID value to update
            **kwargs: Field values to update

        Returns:
            Updated model instance or None if not found
        """
        query = select(self.model).where(self.model.id == id_value)
        result = await self.session.execute(query)
        instance = result.scalar_one_or_none()

        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

            await self.session.flush()  # Send to DB but don't commit
            await self.session.refresh(instance)

        return instance

    async def delete(self, id_value: Any) -> bool:
        """
        Delete a record by ID.

        Args:
            id_value: The ID value to delete

        Returns:
            True if record was deleted, False if not found
        """
        query = select(self.model).where(self.model.id == id_value)
        result = await self.session.execute(query)
        instance = result.scalar_one_or_none()

        if instance:
            await self.session.delete(instance)
            await self.session.flush()  # Send to DB but don't commit
            return True

        return False

    async def search_by_fields(
        self,
        search_fields: dict[str, Any],
        fuzzy: bool = False,
        limit: int | None = None,
    ) -> list[ModelType]:
        """
        Search records by multiple fields.

        Args:
            search_fields: Dictionary of field names and values to search for
            fuzzy: Whether to use fuzzy (ILIKE) matching for string fields
            limit: Maximum number of records to return

        Returns:
            List of matching model instances
        """
        query = select(self.model)
        conditions = []

        for field_name, value in search_fields.items():
            if hasattr(self.model, field_name) and value is not None:
                field = getattr(self.model, field_name)

                if fuzzy and isinstance(value, str):
                    # Use ILIKE for fuzzy string matching
                    conditions.append(field.ilike(f"%{value}%"))
                else:
                    # Exact match
                    conditions.append(field == value)

        if conditions:
            query = query.where(and_(*conditions))

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def search_by_any_field(
        self,
        search_fields: dict[str, Any],
        fuzzy: bool = False,
        limit: int | None = None,
    ) -> list[ModelType]:
        """
        Search records where ANY of the specified fields match (OR condition).

        Args:
            search_fields: Dictionary of field names and values to search for
            fuzzy: Whether to use fuzzy (ILIKE) matching for string fields
            limit: Maximum number of records to return

        Returns:
            List of matching model instances
        """
        query = select(self.model)
        conditions = []

        for field_name, value in search_fields.items():
            if hasattr(self.model, field_name) and value is not None:
                field = getattr(self.model, field_name)

                if fuzzy and isinstance(value, str):
                    # Use ILIKE for fuzzy string matching
                    conditions.append(field.ilike(f"%{value}%"))
                else:
                    # Exact match
                    conditions.append(field == value)

        if conditions:
            query = query.where(or_(*conditions))

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def execute_raw_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[Any]:
        """
        Execute a raw SQL query.

        Args:
            query: Raw SQL query string
            params: Optional parameters for the query

        Returns:
            List of result rows
        """
        result = await self.session.execute(text(query), params or {})
        return result.fetchall()

    async def count(self, search_fields: dict[str, Any] | None = None) -> int:
        """
        Count records with optional filtering.

        Args:
            search_fields: Optional dictionary of field names and values to filter by

        Returns:
            Number of matching records
        """
        query = select(self.model)

        if search_fields:
            conditions = []
            for field_name, value in search_fields.items():
                if hasattr(self.model, field_name) and value is not None:
                    field = getattr(self.model, field_name)
                    conditions.append(field == value)

            if conditions:
                query = query.where(and_(*conditions))

        result = await self.session.execute(query)
        return len(result.scalars().all())


class MultiTableRepository(ABC):
    """
    Repository for operations that span multiple tables.
    Used for complex searches across different model types.
    """

    def __init__(self, session: AsyncSession):
        """Initialize multi-table repository with session."""
        self.session = session

    @abstractmethod
    async def search_across_tables(
        self, search_criteria: dict[str, Any]
    ) -> dict[str, list[Any]]:
        """
        Search across multiple tables based on criteria.
        Must be implemented by concrete repositories.

        Args:
            search_criteria: Dictionary containing search parameters

        Returns:
            Dictionary with table names as keys and lists of results as values
        """
        pass
