from datetime import date

from pydantic import BaseModel


class DrugProductsBase(BaseModel):
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
    pass


class DrugProductsUpdate(BaseModel):
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
    class Config:
        from_attributes = True
