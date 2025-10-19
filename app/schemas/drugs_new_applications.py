
from pydantic import BaseModel


class DrugsNewApplicationsBase(BaseModel):
    document_tracking_number: str
    applicant_company: str | None = None
    brand_name: str | None = None
    generic_name: str | None = None
    dosage_strength: str | None = None
    dosage_form: str | None = None
    packaging: str | None = None
    pharmacologic_category: str | None = None
    application_type: str | None = None


class DrugsNewApplicationsCreate(DrugsNewApplicationsBase):
    pass


class DrugsNewApplicationsUpdate(BaseModel):
    applicant_company: str | None = None
    brand_name: str | None = None
    generic_name: str | None = None
    dosage_strength: str | None = None
    dosage_form: str | None = None
    packaging: str | None = None
    pharmacologic_category: str | None = None
    application_type: str | None = None


class DrugsNewApplicationsResponse(DrugsNewApplicationsBase):
    class Config:
        from_attributes = True
