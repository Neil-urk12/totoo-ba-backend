"""Pydantic schemas for new drug applications.

Defines validation schemas for new drug application data including
base, create, update, and response models.
"""
from pydantic import BaseModel


class DrugsNewApplicationsBase(BaseModel):
    """Base schema for new drug applications with all common fields.

    Attributes:
        document_tracking_number: Unique tracking number for the application.
        applicant_company: Name of the company submitting the application.
        brand_name: Brand name of the drug being applied for.
        generic_name: Generic/scientific name of the drug.
        dosage_strength: Strength/concentration of the drug.
        dosage_form: Form of the drug (tablet, capsule, liquid, etc.).
        packaging: Packaging information.
        pharmacologic_category: Pharmacological category of the drug.
        application_type: Type of application being submitted.
    """
    document_tracking_number: str
    applicant_company: str | None = None
    brand_name: str | None = None
    generic_name: str | None = None
    dosage_strength: str | None = None
    dosage_form: str | None = None
    packaging: str | None = None
    pharmacologic_category: str | None = None
    application_type: str | None = None


class DrugsNewApplicationsCreate(DrugsNewApplicationsBase):
    """Schema for creating a new drug application record.

    Inherits all fields from DrugsNewApplicationsBase.
    """
    pass


class DrugsNewApplicationsUpdate(BaseModel):
    """Schema for updating an existing drug application record.

    All fields are optional to allow partial updates.

    Attributes:
        applicant_company: Name of the company submitting the application.
        brand_name: Brand name of the drug being applied for.
        generic_name: Generic/scientific name of the drug.
        dosage_strength: Strength/concentration of the drug.
        dosage_form: Form of the drug (tablet, capsule, liquid, etc.).
        packaging: Packaging information.
        pharmacologic_category: Pharmacological category of the drug.
        application_type: Type of application being submitted.
    """
    applicant_company: str | None = None
    brand_name: str | None = None
    generic_name: str | None = None
    dosage_strength: str | None = None
    dosage_form: str | None = None
    packaging: str | None = None
    pharmacologic_category: str | None = None
    application_type: str | None = None


class DrugsNewApplicationsResponse(DrugsNewApplicationsBase):
    """Schema for new drug application API responses.

    Inherits all fields from DrugsNewApplicationsBase and enables ORM mode
    for automatic conversion from SQLAlchemy models.
    """
    class Config:
        from_attributes = True
