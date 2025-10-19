from datetime import date

from pydantic import BaseModel


class CosmeticIndustryBase(BaseModel):
    license_number: str
    name_of_establishment: str
    owner: str
    address: str
    region: str
    activity: str
    issuance_date: date
    expiry_date: date


class CosmeticIndustryCreate(CosmeticIndustryBase):
    pass


class CosmeticIndustryUpdate(BaseModel):
    name_of_establishment: str | None = None
    owner: str | None = None
    address: str | None = None
    region: str | None = None
    activity: str | None = None
    issuance_date: date | None = None
    expiry_date: date | None = None


class CosmeticIndustryResponse(CosmeticIndustryBase):
    class Config:
        from_attributes = True
