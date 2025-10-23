"""Pydantic schemas for food products.

Defines validation schemas for food product data including base, create,
update, and response models.
"""
from datetime import date

from pydantic import BaseModel


class FoodProductsBase(BaseModel):
    """Base schema for food products with all common fields.

    Attributes:
        registration_number: Unique registration identifier for the food product.
        company_name: Name of the company producing the food product.
        product_name: Name of the food product.
        brand_name: Brand name of the food product.
        type_of_product: Type/category of the food product.
        issuance_date: Date when the registration was issued.
        expiry_date: Date when the registration expires.
    """
    registration_number: str
    company_name: str | None = None
    product_name: str | None = None
    brand_name: str | None = None
    type_of_product: str | None = None
    issuance_date: date | None = None
    expiry_date: date | None = None


class FoodProductsCreate(FoodProductsBase):
    """Schema for creating a new food product record.

    Inherits all fields from FoodProductsBase.
    """
    pass


class FoodProductsUpdate(BaseModel):
    """Schema for updating an existing food product record.

    All fields are optional to allow partial updates.

    Attributes:
        company_name: Name of the company producing the food product.
        product_name: Name of the food product.
        brand_name: Brand name of the food product.
        type_of_product: Type/category of the food product.
        issuance_date: Date when the registration was issued.
        expiry_date: Date when the registration expires.
    """
    company_name: str | None = None
    product_name: str | None = None
    brand_name: str | None = None
    type_of_product: str | None = None
    issuance_date: date | None = None
    expiry_date: date | None = None


class FoodProductsResponse(FoodProductsBase):
    """Schema for food product API responses.

    Inherits all fields from FoodProductsBase and enables ORM mode
    for automatic conversion from SQLAlchemy models.
    """
    class Config:
        from_attributes = True
