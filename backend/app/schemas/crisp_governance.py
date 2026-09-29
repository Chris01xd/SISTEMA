import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.crisp_governance import DocumentType


class CrispDocumentOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    phase_id: uuid.UUID | None
    name: str
    document_type: DocumentType
    storage_path: str
    description: str
    uploaded_by_id: uuid.UUID | None
    model_config = {"from_attributes": True}


class CrispAuditLogOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID | None
    action: str
    occurred_at: datetime
    project_id: uuid.UUID | None
    phase_id: uuid.UUID | None
    entity_type: str
    entity_id: str
    old_value: dict
    new_value: dict
    ip_address: str | None
    model_config = {"from_attributes": True}
