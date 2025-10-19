from pydantic import BaseModel
from datetime import date
from typing import Optional


class FoodProductsBase(BaseModel):
    registration_number: str
    company_name: Optional[str] = None
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    type_of_product: Optional[str] = None
    issuance_date: Optional[date] = None
    expiry_date: Optional[date] = None


class FoodProductsCreate(FoodProductsBase):
    pass


class FoodProductsUpdate(BaseModel):
    company_name: Optional[str] = None
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    type_of_product: Optional[str] = None
    issuance_date: Optional[date] = None
    expiry_date: Optional[date] = None


class FoodProductsResponse(FoodProductsBase):
    class Config:
        from_attributes = True
