import uuid
from datetime import date

from pydantic import BaseModel

from app.models.crisp_project import ProjectMemberRole, ProjectOverallStatus


class CrispProjectCreate(BaseModel):
    name: str
    description: str = ""
    digital_twin_id: uuid.UUID | None = None
    responsible_id: uuid.UUID | None = None
    start_date: date
    estimated_end_date: date | None = None
    loading_unit_ids: list[uuid.UUID] = []
    member_ids: list[tuple[uuid.UUID, ProjectMemberRole]] = []


class CrispProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    responsible_id: uuid.UUID | None = None
    estimated_end_date: date | None = None
    overall_status: ProjectOverallStatus | None = None


class CrispProjectOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    digital_twin_id: uuid.UUID | None
    responsible_id: uuid.UUID | None
    start_date: date
    estimated_end_date: date | None
    actual_end_date: date | None
    current_phase_number: int
    overall_status: ProjectOverallStatus
    progress_percent: float
    status: str
    model_config = {"from_attributes": True}


class CrispProjectMemberOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    role_in_project: ProjectMemberRole
    model_config = {"from_attributes": True}
