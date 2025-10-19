from datetime import date

from sqlalchemy import Date, String
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class FoodProducts(Base):
    """
    SQLAlchemy model for the Food Products table in Supabase database.

    Represents food products with their registration and product information.
    """

    __tablename__ = "food_products"

    registration_number: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    company_name: Mapped[str] = mapped_column(String, nullable=True)
    product_name: Mapped[str] = mapped_column(String, nullable=True)
    brand_name: Mapped[str] = mapped_column(String, nullable=True)
    type_of_product: Mapped[str] = mapped_column(String, nullable=True)
    issuance_date: Mapped[date] = mapped_column(Date, nullable=True)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=True)

    # Full-text search vector column (generated in database)
    # Note: This is a GENERATED ALWAYS AS column - it's computed automatically
    search_vector = mapped_column(
        TSVECTOR,
        nullable=True,
        # Don't include this in INSERT/UPDATE operations
        insert_default=None,
    )

    def __repr__(self) -> str:
        return f"<FoodProducts(registration_number='{self.registration_number}', product_name='{self.product_name}')>"
