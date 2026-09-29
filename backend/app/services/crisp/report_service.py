"""Assembles the final CRISP-DM project report from live DB rows and writes
it via app/reports/crisp_report_generator.py, registering the output as a
`crisp_documents` row (same idea as app/services/reports_service.py does for
`reports`)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.crisp_deployment import ModelDeployment
from app.models.crisp_governance import CrispDocument, DocumentType
from app.models.crisp_modeling import MlExperiment
from app.models.crisp_phase import PHASE_NAMES, CrispPhase
from app.models.user import User
from app.reports.crisp_report_generator import generate_crisp_excel_report, generate_crisp_pdf_report
from app.services.crisp.audit_service import record_crisp_audit
from app.services.crisp.modeling_service import get_experiment_metrics
from app.services.crisp.project_service import get_project


def _gather_report_data(db: Session, project_id: uuid.UUID) -> dict:
    project = get_project(db, project_id)
    responsible = db.get(User, project.responsible_id) if project.responsible_id else None

    phases = db.query(CrispPhase).filter(CrispPhase.project_id == project_id).order_by(CrispPhase.phase_number).all()
    phase_1 = next((p for p in phases if p.phase_number == 1), None)
    kpis = (phase_1.phase_data or {}).get("kpis", []) if phase_1 else []

    experiments = db.query(MlExperiment).filter(MlExperiment.project_id == project_id).all()
    experiment_rows = [
        {
            "algorithm": e.algorithm.value,
            "objective": e.objective.value,
            "status": e.status.value,
            "metrics": {m.metric_name: m.metric_value for m in get_experiment_metrics(db, e.id)},
        }
        for e in experiments
    ]

    deployments = db.query(ModelDeployment).filter(ModelDeployment.project_id == project_id).all()
    deployment_rows = [
        {
            "version": d.version,
            "service_status": d.service_status.value,
            "api_endpoint": d.api_endpoint,
            "next_retraining_date": str(d.next_retraining_date) if d.next_retraining_date else None,
        }
        for d in deployments
    ]

    return {
        "project": {
            "name": project.name,
            "responsible": responsible.full_name if responsible else None,
            "overall_status": project.overall_status.value,
            "progress_percent": project.progress_percent,
        },
        "phases": [
            {
                "phase_number": p.phase_number,
                "name": PHASE_NAMES[p.phase_number],
                "status": p.status.value,
                "approved_at": str(p.approved_at) if p.approved_at else None,
            }
            for p in phases
        ],
        "kpis": kpis,
        "experiments": experiment_rows,
        "deployments": deployment_rows,
    }


def generate_final_report(db: Session, project_id: uuid.UUID, fmt: str, actor_id: uuid.UUID) -> CrispDocument:
    if fmt not in {"pdf", "xlsx"}:
        raise HTTPException(status_code=400, detail="fmt must be 'pdf' or 'xlsx'")
    data = _gather_report_data(db, project_id)

    output_dir = Path(settings.REPORT_OUTPUT_DIR)
    file_name = f"crisp_report_{project_id}_{int(datetime.now(timezone.utc).timestamp())}.{fmt}"
    output_path = output_dir / file_name

    if fmt == "pdf":
        generate_crisp_pdf_report(output_path, data["project"], data["phases"], data["kpis"], data["experiments"], data["deployments"])
        doc_type = DocumentType.FINAL_REPORT_PDF
    else:
        generate_crisp_excel_report(output_path, data["project"], data["phases"], data["kpis"], data["experiments"], data["deployments"])
        doc_type = DocumentType.FINAL_REPORT_XLSX

    document = CrispDocument(
        project_id=project_id,
        name=file_name,
        document_type=doc_type,
        storage_path=str(output_path),
        description=f"Informe final generado automáticamente ({fmt.upper()}).",
        uploaded_by_id=actor_id,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    record_crisp_audit(
        db, user_id=actor_id, action="GENERATE_FINAL_REPORT", entity_type="CrispDocument", entity_id=str(document.id),
        project_id=project_id, new_value={"format": fmt},
    )
    return document
