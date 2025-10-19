from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DrugsNewApplications(Base):
    """
    SQLAlchemy model for the Drugs New Applications table in Supabase database.

    Represents new drug applications with their tracking information.
    """

    __tablename__ = "Drugs New Applications"

    document_tracking_number: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    applicant_company: Mapped[str] = mapped_column(String, nullable=True)
    brand_name: Mapped[str] = mapped_column(String, nullable=True)
    generic_name: Mapped[str] = mapped_column(String, nullable=True)
    dosage_strength: Mapped[str] = mapped_column(String, nullable=True)
    dosage_form: Mapped[str] = mapped_column(String, nullable=True)
    packaging: Mapped[str] = mapped_column(String, nullable=True)
    pharmacologic_category: Mapped[str] = mapped_column(String, nullable=True)
    application_type: Mapped[str] = mapped_column(String, nullable=True)

    def __repr__(self) -> str:
        return f"<DrugsNewApplications(document_tracking_number='{self.document_tracking_number}', brand_name='{self.brand_name}')>"
