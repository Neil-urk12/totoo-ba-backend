"""Pydantic schemas for food industry establishments.

Defines validation schemas for food industry establishment data including
base, create, update, and response models.
"""
from datetime import date

from pydantic import BaseModel


class FoodIndustryBase(BaseModel):
    """Base schema for food industry establishments with all common fields.
    
    Attributes:
        license_number: Unique license identifier for the establishment.
        name_of_establishment: Name of the food industry establishment.
        owner: Owner of the establishment.
        address: Physical address of the establishment.
        region: Geographic region where the establishment is located.
        activity: Type of activity/operation performed by the establishment.
        issuance_date: Date when the license was issued.
        expiry_date: Date when the license expires.
    """
    license_number: str
    name_of_establishment: str
    owner: str
    address: str
    region: str
    activity: str
    issuance_date: date
    expiry_date: date


class FoodIndustryCreate(FoodIndustryBase):
    """Schema for creating a new food industry establishment record.
    
    Inherits all fields from FoodIndustryBase.
    """
    pass


class FoodIndustryUpdate(BaseModel):
    """Schema for updating an existing food industry establishment record.
    
    All fields are optional to allow partial updates.
    
    Attributes:
        name_of_establishment: Name of the food industry establishment.
        owner: Owner of the establishment.
        address: Physical address of the establishment.
        region: Geographic region where the establishment is located.
        activity: Type of activity/operation performed by the establishment.
        issuance_date: Date when the license was issued.
        expiry_date: Date when the license expires.
    """
    name_of_establishment: str | None = None
    owner: str | None = None
    address: str | None = None
    region: str | None = None
    activity: str | None = None
    issuance_date: date | None = None
    expiry_date: date | None = None


class FoodIndustryResponse(FoodIndustryBase):
    """Schema for food industry establishment API responses.
    
    Inherits all fields from FoodIndustryBase and enables ORM mode
    for automatic conversion from SQLAlchemy models.
    """
    class Config:
        from_attributes = True
