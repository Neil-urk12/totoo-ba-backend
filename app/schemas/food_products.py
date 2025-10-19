from datetime import date

from pydantic import BaseModel


class FoodProductsBase(BaseModel):
    registration_number: str
    company_name: str | None = None
    product_name: str | None = None
    brand_name: str | None = None
    type_of_product: str | None = None
    issuance_date: date | None = None
    expiry_date: date | None = None


class FoodProductsCreate(FoodProductsBase):
    pass


class FoodProductsUpdate(BaseModel):
    company_name: str | None = None
    product_name: str | None = None
    brand_name: str | None = None
    type_of_product: str | None = None
    issuance_date: date | None = None
    expiry_date: date | None = None


class FoodProductsResponse(FoodProductsBase):
    class Config:
        from_attributes = True
