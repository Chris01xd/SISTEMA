"""Ties simulation results in Postgres to the DOCX/PDF/XLSX generators."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.emergency import EmergencyEvent
from app.models.hospital import Hospital
from app.models.report import Report, ReportFormat
from app.models.simulation import Simulation, SimulationResult
from app.reports.docx_generator import generate_docx_report
from app.reports.excel_generator import generate_excel_report
from app.reports.pdf_generator import generate_pdf_report


def _summary_metrics(results: list[SimulationResult]) -> dict:
    if not results:
        return {}
    last = results[-1].metrics
    peak_saturation = max(r.metrics.get("saturation_index", 0) for r in results)
    peak_waiting = max(r.metrics.get("patients_waiting", 0) for r in results)
    peak_occupied = max(r.metrics.get("beds_occupied", 0) for r in results)
    beds_total = last.get("beds_total", 0)
    return {
        "Total de pacientes atendidos": last.get("total_patients_seen", 0),
        "Fallecidos": last.get("patients_deceased", 0),
        "Dados de alta": last.get("patients_discharged", 0),
        "Índice de saturación máximo": round(peak_saturation, 3),
        "Pico de pacientes en espera": peak_waiting,
        # Reporting only the *final* tick's occupancy reads as "0 camas
        # ocupadas" for any horizon that outlasts the crisis (everyone's
        # already discharged by then) even when the hospital was genuinely
        # full at the peak — show both so the after-action picture is honest.
        "Pico de camas ocupadas / totales": f"{peak_occupied}/{beds_total}",
        "Camas ocupadas al final / totales": f"{last.get('beds_occupied', 0)}/{beds_total}",
    }


def generate_report(db: Session, simulation_id: uuid.UUID, author_id: uuid.UUID, fmt: ReportFormat) -> Report:
    sim = db.get(Simulation, simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    event = db.get(EmergencyEvent, sim.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Emergency event for this simulation was not found")
    hospital = db.get(Hospital, event.hospital_id)
    if hospital is None:
        raise HTTPException(status_code=404, detail="Hospital for this simulation was not found")
    results = (
        db.query(SimulationResult)
        .filter(SimulationResult.simulation_id == simulation_id)
        .order_by(SimulationResult.tick)
        .all()
    )
    timeseries = [
        {"tick": r.tick, **{k: v for k, v in r.metrics.items() if not isinstance(v, dict)}} for r in results
    ]
    summary = _summary_metrics(results)
    recommendations = sim.config.get("recommendations", []) if sim.config else []

    out_dir = Path(settings.REPORT_OUTPUT_DIR)
    filename = f"report_{simulation_id}.{fmt.value.lower()}"
    output_path = out_dir / filename

    if fmt == ReportFormat.DOCX:
        generate_docx_report(
            output_path, hospital.name, event.name, str(simulation_id), summary, timeseries, recommendations
        )
    elif fmt == ReportFormat.PDF:
        generate_pdf_report(output_path, hospital.name, event.name, str(simulation_id), summary, timeseries)
    elif fmt == ReportFormat.XLSX:
        generate_excel_report(output_path, summary, timeseries)

    report = Report(
        simulation_id=simulation_id,
        author_id=author_id,
        format=fmt,
        storage_path=str(output_path),
        generated_at=datetime.now(timezone.utc),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report
