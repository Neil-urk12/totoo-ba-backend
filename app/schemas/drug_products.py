from pydantic import BaseModel
from datetime import date
from typing import Optional


class DrugProductsBase(BaseModel):
    registration_number: str
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


class DrugProductsCreate(DrugProductsBase):
    pass


class DrugProductsUpdate(BaseModel):
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


class DrugProductsResponse(DrugProductsBase):
    class Config:
        from_attributes = True
