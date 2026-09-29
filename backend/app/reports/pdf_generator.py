"""PDF technical report generation (ReportLab) — charts, tables, indicators."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.reports.labels import STATUS_COLORS_HEX, column_label, format_value

PAGE_SIZE = landscape(A4)
PAGE_MARGIN = 1.3 * cm
USABLE_WIDTH = PAGE_SIZE[0] - 2 * PAGE_MARGIN

# The time series table can have a dozen-plus columns (tick, census fields,
# saturation, status, fatigue...) — splitting the width evenly made every
# column too narrow to hold a word, so ReportLab wrapped mid-word ("Patient
# / s Waiti / ng"). These two are inherently short, so give them a fixed
# width and let every other column share what's left.
NARROW_COLUMN_WIDTHS = {"tick": 1.1 * cm, "status_band": 1.8 * cm}


def _column_widths(cols: list[str]) -> list[float]:
    reserved = sum(NARROW_COLUMN_WIDTHS.get(c, 0) for c in cols)
    wide_cols = [c for c in cols if c not in NARROW_COLUMN_WIDTHS]
    per_wide_col = (USABLE_WIDTH - reserved) / max(len(wide_cols), 1)
    return [NARROW_COLUMN_WIDTHS.get(c, per_wide_col) for c in cols]


def generate_pdf_report(
    output_path: Path,
    hospital_name: str,
    event_name: str,
    simulation_id: str,
    summary_metrics: dict,
    timeseries_sample: list[dict],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=8, leading=10, alignment=TA_CENTER)
    value_style = ParagraphStyle("CellRight", parent=cell_style, alignment=TA_RIGHT)
    header_style = ParagraphStyle(
        "CellHeader", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold"
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=PAGE_SIZE,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
    )
    story = []

    story.append(Paragraph("Hospital Digital Twin — Reporte Técnico", styles["Title"]))
    story.append(Paragraph(f"Hospital: {hospital_name} | Evento: {event_name}", styles["Normal"]))
    story.append(Paragraph(f"ID de Simulación: {simulation_id}", styles["Normal"]))
    story.append(Paragraph(f"Generado: {datetime.now(timezone.utc).isoformat()} UTC", styles["Normal"]))
    story.append(Spacer(1, 1 * cm))

    story.append(Paragraph("Indicadores Clave", styles["Heading2"]))
    metric_rows = [["Métrica", "Valor"]] + [
        [str(k), format_value(k, v)] for k, v in summary_metrics.items()
    ]
    metric_table = Table(metric_rows, colWidths=[USABLE_WIDTH * 0.7, USABLE_WIDTH * 0.3], hAlign="LEFT")
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 1 * cm))

    if timeseries_sample:
        story.append(Paragraph("Serie Temporal de la Simulación (muestra)", styles["Heading2"]))
        cols = list(timeseries_sample[0].keys())
        col_widths = _column_widths(cols)
        ts_cell_style = ParagraphStyle("TsCell", parent=value_style, fontSize=7, leading=8)
        ts_header_style = ParagraphStyle("TsHeader", parent=header_style, fontSize=7, leading=8)
        status_cell_styles = {
            status: ParagraphStyle(
                f"TsStatus{status}", parent=ts_cell_style, textColor=colors.white, fontName="Helvetica-Bold"
            )
            for status in STATUS_COLORS_HEX
        }
        status_col = cols.index("status_band") if "status_band" in cols else None

        # Every cell is a Paragraph (not a raw string) so ReportLab wraps long
        # headers/values to fit their column instead of silently widening the
        # table past the page margins — that overflow was the original bug.
        header_row = [Paragraph(column_label(c), ts_header_style) for c in cols]
        rows_data = timeseries_sample[:40]
        data_rows = []
        for row in rows_data:
            cells = []
            for c in cols:
                raw = row.get(c, "")
                style = status_cell_styles.get(raw, ts_cell_style) if c == "status_band" else ts_cell_style
                cells.append(Paragraph(format_value(c, raw), style))
            data_rows.append(cells)

        table_style_commands = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]
        if status_col is not None:
            for row_idx, row in enumerate(rows_data, start=1):
                hex_color = STATUS_COLORS_HEX.get(row.get("status_band"))
                if hex_color:
                    table_style_commands.append(
                        ("BACKGROUND", (status_col, row_idx), (status_col, row_idx), colors.HexColor(f"#{hex_color}"))
                    )

        ts_table = Table(
            [header_row] + data_rows,
            colWidths=col_widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        ts_table.setStyle(TableStyle(table_style_commands))
        story.append(ts_table)

    doc.build(story)
    return output_path
