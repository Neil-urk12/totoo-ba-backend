from pydantic import BaseModel
from datetime import date
from typing import Optional


class ProductBase(BaseModel):
    """Base schema for Product with required fields"""

    registration_number: str
    product_type: Optional[str] = None
    product_name: Optional[str] = None
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    dosage_strength: Optional[str] = None
    dosage_form: Optional[str] = None
    classification: Optional[str] = None
    packaging: Optional[str] = None
    pharmacologic_category: Optional[str] = None
    manufacturer: Optional[str] = None
    country_of_origin: Optional[str] = None
    trader: Optional[str] = None
    importer: Optional[str] = None
    distributor: Optional[str] = None
    application_type: Optional[str] = None
    issuance_date: Optional[date] = None
    expiry_date: Optional[date] = None


class ProductCreate(ProductBase):
    """Schema for creating a new Product record"""

    pass


class ProductUpdate(BaseModel):
    """Schema for updating an existing Product record"""

    product_name: Optional[str] = None
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    dosage_strength: Optional[str] = None
    dosage_form: Optional[str] = None
    classification: Optional[str] = None
    packaging: Optional[str] = None
    pharmacologic_category: Optional[str] = None
    manufacturer: Optional[str] = None
    country_of_origin: Optional[str] = None
    trader: Optional[str] = None
    importer: Optional[str] = None
    distributor: Optional[str] = None
    application_type: Optional[str] = None
    issuance_date: Optional[date] = None
    expiry_date: Optional[date] = None
    product_type: Optional[str] = None


class Product(ProductBase):
    """Schema for Product with all fields including the primary key"""

    class Config:
        from_attributes = True
