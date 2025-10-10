from sqlalchemy import String, Date
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from datetime import date


class DrugProducts(Base):
    """
    SQLAlchemy model for the Drug Products table in Supabase database.
    
    Represents drug products with their registration and product information.
    """
    __tablename__ = "Drug Products"

    registration_number: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    product_information: Mapped[str] = mapped_column(String, nullable=True)
    generic_name: Mapped[str] = mapped_column(String, nullable=True)
    brand_name: Mapped[str] = mapped_column(String, nullable=True)
    dosage_strength: Mapped[str] = mapped_column(String, nullable=True)
    dosage_form: Mapped[str] = mapped_column(String, nullable=True)
    classification: Mapped[str] = mapped_column(String, nullable=True)
    packaging: Mapped[str] = mapped_column(String, nullable=True)
    pharmacologic_category: Mapped[str] = mapped_column(String, nullable=True)
    manufacturer: Mapped[str] = mapped_column(String, nullable=True)
    country_of_origin: Mapped[str] = mapped_column(String, nullable=True)
    trader: Mapped[str] = mapped_column(String, nullable=True)
    importer: Mapped[str] = mapped_column(String, nullable=True)
    distributor: Mapped[str] = mapped_column(String, nullable=True)
    application_type: Mapped[str] = mapped_column(String, nullable=True)
    issuance_date: Mapped[date] = mapped_column(Date, nullable=True)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=True)

    def __repr__(self) -> str:
        return f"<DrugProducts(registration_number='{self.registration_number}', brand_name='{self.brand_name}')>"