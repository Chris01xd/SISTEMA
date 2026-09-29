from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.authentication.dependencies import require_permission
from app.database.session import get_db
from app.models.user import User
from app.schemas.reports import ReportGenerateRequest, ReportOut
from app.services.reports_service import generate_report

router = APIRouter()


@router.post("/generate", response_model=ReportOut, dependencies=[Depends(require_permission("reports:generate"))])
def generate_report_endpoint(
    payload: ReportGenerateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("reports:generate")),
) -> ReportOut:
    return generate_report(db, payload.simulation_id, user.id, payload.format)


@router.get("/{report_id}/download")
def download_report(report_id: str, db: Session = Depends(get_db)):
    from app.models.report import Report

    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report.storage_path, filename=report.storage_path.split("/")[-1])
