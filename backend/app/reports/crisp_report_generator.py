"""Final CRISP-DM project report (PDF via ReportLab, XLSX via pandas/openpyxl).

A new module rather than reusing app/reports/pdf_generator.py /
excel_generator.py as-is: those generators hard-code the hospital domain
("Hospital Digital Twin — Reporte Técnico" title, hospital_name/event_name/
simulation_id parameters, patient/status_band-specific columns) and aren't
shaped for a six-phase CRISP-DM project. This keeps the same libraries and a
similar dark-header table style, applied to CRISP-DM's own content."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PAGE_SIZE = landscape(A4)
MARGIN = 1.3 * cm
USABLE_WIDTH = PAGE_SIZE[0] - 2 * MARGIN

_TABLE_STYLE = TableStyle(
    [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
)


def _table(rows: list[list[str]], widths: list[float] | None = None) -> Table:
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(_TABLE_STYLE)
    return t


def generate_crisp_pdf_report(
    output_path: Path,
    project: dict,
    phases: list[dict],
    kpis: list[dict],
    experiments: list[dict],
    deployments: list[dict],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(output_path), pagesize=PAGE_SIZE, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN
    )
    story = [
        Paragraph(f"Informe Final CRISP-DM — {project['name']}", styles["Title"]),
        Paragraph(f"Responsable: {project.get('responsible', '—')} | Estado: {project['overall_status']}", styles["Normal"]),
        Paragraph(f"Generado: {datetime.now(timezone.utc).isoformat()} UTC", styles["Normal"]),
        Spacer(1, 0.6 * cm),
    ]

    story.append(Paragraph("Avance por fase CRISP-DM", styles["Heading2"]))
    phase_rows = [["Fase", "Nombre", "Estado", "Aprobada"]] + [
        [str(p["phase_number"]), p["name"], p["status"], "Sí" if p["approved_at"] else "No"] for p in phases
    ]
    story.append(_table(phase_rows))
    story.append(Spacer(1, 0.6 * cm))

    if kpis:
        story.append(Paragraph("Indicadores de negocio: línea base, meta y resultado real", styles["Heading2"]))
        kpi_rows = [["Indicador", "Línea base", "Meta", "Resultado real"]] + [
            [k["indicator"], _fmt(k.get("baseline_value")), _fmt(k.get("target_value")), _fmt(k.get("actual_value")) or "Pendiente (piloto en curso)"]
            for k in kpis
        ]
        story.append(_table(kpi_rows))
        story.append(Spacer(1, 0.6 * cm))

    if experiments:
        story.append(Paragraph("Experimentos y métricas técnicas", styles["Heading2"]))
        exp_rows = [["Algoritmo", "Objetivo", "Estado", "Métricas (test)"]] + [
            [e["algorithm"], e["objective"], e["status"], ", ".join(f"{k}={v}" for k, v in e.get("metrics", {}).items())]
            for e in experiments
        ]
        story.append(_table(exp_rows))
        story.append(Spacer(1, 0.6 * cm))

    if deployments:
        story.append(Paragraph("Despliegues", styles["Heading2"]))
        dep_rows = [["Versión", "Estado", "Endpoint", "Próximo reentrenamiento"]] + [
            [d["version"], d["service_status"], d["api_endpoint"], d.get("next_retraining_date") or "—"] for d in deployments
        ]
        story.append(_table(dep_rows))

    doc.build(story)
    return output_path


def _fmt(value) -> str:
    return "" if value is None else str(value)


def generate_crisp_excel_report(
    output_path: Path,
    project: dict,
    phases: list[dict],
    kpis: list[dict],
    experiments: list[dict],
    deployments: list[dict],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([project]).to_excel(writer, sheet_name="Proyecto", index=False)
        pd.DataFrame(phases).to_excel(writer, sheet_name="Fases", index=False)
        if kpis:
            pd.DataFrame(kpis).to_excel(writer, sheet_name="Indicadores", index=False)
        if experiments:
            pd.DataFrame(experiments).to_excel(writer, sheet_name="Experimentos", index=False)
        if deployments:
            pd.DataFrame(deployments).to_excel(writer, sheet_name="Despliegues", index=False)
    return output_path
