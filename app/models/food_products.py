from sqlalchemy import String, Date
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from datetime import date


class FoodProducts(Base):
    """
    SQLAlchemy model for the Food Products table in Supabase database.
    
    Represents food products with their registration and product information.
    """
    __tablename__ = "food_products"

    registration_number: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    company_name: Mapped[str] = mapped_column(String, nullable=True)
    product_name: Mapped[str] = mapped_column(String, nullable=True)
    brand_name: Mapped[str] = mapped_column(String, nullable=True)
    type_of_product: Mapped[str] = mapped_column(String, nullable=True)
    issuance_date: Mapped[date] = mapped_column(Date, nullable=True)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=True)

    def __repr__(self) -> str:
        return f"<FoodProducts(registration_number='{self.registration_number}', product_name='{self.product_name}')>"