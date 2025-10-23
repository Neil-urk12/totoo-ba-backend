"""Pydantic schemas for drug products.

Defines validation schemas for drug product data including base, create,
update, and response models.
"""
from datetime import date

from pydantic import BaseModel


class DrugProductsBase(BaseModel):
    """Base schema for drug products with all common fields.
    
    Attributes:
        registration_number: Unique registration identifier for the drug product.
        generic_name: Generic/scientific name of the drug.
        brand_name: Commercial brand name of the drug.
        dosage_strength: Strength/concentration of the drug.
        dosage_form: Form of the drug (tablet, capsule, liquid, etc.).
        classification: Drug classification category.
        packaging: Packaging information.
        pharmacologic_category: Pharmacological category of the drug.
        manufacturer: Name of the manufacturer.
        country_of_origin: Country where the drug is manufactured.
        trader: Trading company name.
        importer: Importing company name.
        distributor: Distributing company name.
        application_type: Type of application submitted.
        issuance_date: Date when the registration was issued.
        expiry_date: Date when the registration expires.
    """
    registration_number: str
    generic_name: str | None = None
    brand_name: str | None = None
    dosage_strength: str | None = None
    dosage_form: str | None = None
    classification: str | None = None
    packaging: str | None = None
    pharmacologic_category: str | None = None
    manufacturer: str | None = None
    country_of_origin: str | None = None
    trader: str | None = None
    importer: str | None = None
    distributor: str | None = None
    application_type: str | None = None
    issuance_date: date | None = None
    expiry_date: date | None = None


class DrugProductsCreate(DrugProductsBase):
    """Schema for creating a new drug product record.
    
    Inherits all fields from DrugProductsBase.
    """
    pass


class DrugProductsUpdate(BaseModel):
    """Schema for updating an existing drug product record.
    
    All fields are optional to allow partial updates.
    
    Attributes:
        generic_name: Generic/scientific name of the drug.
        brand_name: Commercial brand name of the drug.
        dosage_strength: Strength/concentration of the drug.
        dosage_form: Form of the drug (tablet, capsule, liquid, etc.).
        classification: Drug classification category.
        packaging: Packaging information.
        pharmacologic_category: Pharmacological category of the drug.
        manufacturer: Name of the manufacturer.
        country_of_origin: Country where the drug is manufactured.
        trader: Trading company name.
        importer: Importing company name.
        distributor: Distributing company name.
        application_type: Type of application submitted.
        issuance_date: Date when the registration was issued.
        expiry_date: Date when the registration expires.
    """
    generic_name: str | None = None
    brand_name: str | None = None
    dosage_strength: str | None = None
    dosage_form: str | None = None
    classification: str | None = None
    packaging: str | None = None
    pharmacologic_category: str | None = None
    manufacturer: str | None = None
    country_of_origin: str | None = None
    trader: str | None = None
    importer: str | None = None
    distributor: str | None = None
    application_type: str | None = None
    issuance_date: date | None = None
    expiry_date: date | None = None


class DrugProductsResponse(DrugProductsBase):
    """Schema for drug product API responses.
    
    Inherits all fields from DrugProductsBase and enables ORM mode
    for automatic conversion from SQLAlchemy models.
    """
    class Config:
        from_attributes = True
