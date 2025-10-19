from pydantic import BaseModel
from datetime import date
from typing import Optional


class DrugIndustryBase(BaseModel):
    license_number: str
    name_of_establishment: str
    owner: str
    address: str
    region: str
    activity: str
    issuance_date: date
    expiry_date: date


class DrugIndustryCreate(DrugIndustryBase):
    pass


class DrugIndustryUpdate(BaseModel):
    name_of_establishment: Optional[str] = None
    owner: Optional[str] = None
    address: Optional[str] = None
    region: Optional[str] = None
    activity: Optional[str] = None
    issuance_date: Optional[date] = None
    expiry_date: Optional[date] = None


class DrugIndustryResponse(DrugIndustryBase):
    class Config:
        from_attributes = True
