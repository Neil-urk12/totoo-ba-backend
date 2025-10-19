from sqlalchemy import String, Date
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from datetime import date


class CosmeticIndustry(Base):
    """
    SQLAlchemy model for the Cosmetic Industry table in Supabase database.

    Represents cosmetic industry establishments with their licensing information.
    """

    __tablename__ = "Cosmetic Industry"

    license_number: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    name_of_establishment: Mapped[str] = mapped_column(String, nullable=False)
    owner: Mapped[str] = mapped_column(String, nullable=False)
    address: Mapped[str] = mapped_column(String, nullable=False)
    region: Mapped[str] = mapped_column(String, nullable=False)
    activity: Mapped[str] = mapped_column(String, nullable=False)
    issuance_date: Mapped[date] = mapped_column(Date, nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False)

    def __repr__(self) -> str:
        return f"<CosmeticIndustry(license_number='{self.license_number}', name='{self.name_of_establishment}')>"
