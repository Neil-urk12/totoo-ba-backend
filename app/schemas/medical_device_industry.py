from datetime import date

from pydantic import BaseModel


class MedicalDeviceIndustryBase(BaseModel):
    license_number: str
    name_of_establishment: str
    owner: str
    address: str
    region: str
    activity: str
    issuance_date: date
    expiry_date: date


class MedicalDeviceIndustryCreate(MedicalDeviceIndustryBase):
    pass


class MedicalDeviceIndustryUpdate(BaseModel):
    name_of_establishment: str | None = None
    owner: str | None = None
    address: str | None = None
    region: str | None = None
    activity: str | None = None
    issuance_date: date | None = None
    expiry_date: date | None = None


class MedicalDeviceIndustryResponse(MedicalDeviceIndustryBase):
    class Config:
        from_attributes = True
