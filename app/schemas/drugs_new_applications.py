from pydantic import BaseModel
from typing import Optional


class DrugsNewApplicationsBase(BaseModel):
    document_tracking_number: str
    applicant_company: Optional[str] = None
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    dosage_strength: Optional[str] = None
    dosage_form: Optional[str] = None
    packaging: Optional[str] = None
    pharmacologic_category: Optional[str] = None
    application_type: Optional[str] = None


class DrugsNewApplicationsCreate(DrugsNewApplicationsBase):
    pass


class DrugsNewApplicationsUpdate(BaseModel):
    applicant_company: Optional[str] = None
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    dosage_strength: Optional[str] = None
    dosage_form: Optional[str] = None
    packaging: Optional[str] = None
    pharmacologic_category: Optional[str] = None
    application_type: Optional[str] = None


class DrugsNewApplicationsResponse(DrugsNewApplicationsBase):
    class Config:
        from_attributes = True
