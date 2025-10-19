from datetime import date

from pydantic import BaseModel


class ProductBase(BaseModel):
    """Base schema for Product with required fields"""

    registration_number: str
    product_type: str | None = None
    product_name: str | None = None
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


class ProductCreate(ProductBase):
    """Schema for creating a new Product record"""

    pass


class ProductUpdate(BaseModel):
    """Schema for updating an existing Product record"""

    product_name: str | None = None
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
    product_type: str | None = None


class Product(ProductBase):
    """Schema for Product with all fields including the primary key"""

    class Config:
        from_attributes = True
