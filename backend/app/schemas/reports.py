import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.report import ReportFormat


class ReportGenerateRequest(BaseModel):
    simulation_id: uuid.UUID
    format: ReportFormat


class ReportOut(BaseModel):
    id: uuid.UUID
    simulation_id: uuid.UUID
    format: ReportFormat
    storage_path: str
    generated_at: datetime
    model_config = {"from_attributes": True}
