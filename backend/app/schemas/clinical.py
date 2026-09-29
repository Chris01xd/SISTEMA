import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.clinical import PatientStatus, SeverityLevel, Shift, StaffRole


class PatientCreate(BaseModel):
    age: int
    sex: str
    diagnosis: str = ""
    severity_level: SeverityLevel
    max_wait_time_minutes: int = 60
    resource_needs: dict = {}
    event_id: uuid.UUID | None = None


class PatientOut(BaseModel):
    id: uuid.UUID
    age: int
    sex: str
    diagnosis: str
    severity_level: SeverityLevel
    triage_priority: int
    arrival_time: datetime
    status: PatientStatus
    room_id: uuid.UUID | None
    model_config = {"from_attributes": True}


class MedicalStaffOut(BaseModel):
    id: uuid.UUID
    full_name: str
    role: StaffRole
    specialty: str
    shift: Shift
    assigned_area: str
    capacity: int
    workload_score: float
    is_available: bool
    model_config = {"from_attributes": True}
