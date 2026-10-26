"""Report export buttons, reusing backend/app/reports/ml_analysis_report.py as-is
(pure functions over the same `output` dict shown on screen — no FastAPI/DB involved)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from lib.paths import bootstrap

bootstrap()

from backend.app.reports.ml_analysis_report import L, generate_ml_pdf, generate_ml_xlsx, report_tables  # noqa: E402
from lib.charts import all_figures  # noqa: E402
from lib.export import build_assets_zip, figure_bytes, slug  # noqa: E402


def _build(out: dict[str, Any], tmp: Path) -> dict[str, bytes]:
    """PDF (with HD chart annex), Excel and a ZIP with every chart (PNG/SVG/HTML), table (CSV/XLSX), the PDF and the Excel."""
    figs = all_figures(out)
    pngs = {name: png for name, fig in figs.items() if (png := figure_bytes(fig, "png"))}
    pdf = generate_ml_pdf(out, tmp / f"crisp_dm_{out['run_id']}.pdf", engine="streamlit", figures=pngs)
    xlsx = generate_ml_xlsx(out, tmp / f"crisp_dm_{out['run_id']}.xlsx")
    labels = L.get(out.get("language", "es"), L["es"])
    tables = {labels[title]: df for _, title, _, df in report_tables(out)}
    files = {pdf.name: pdf.read_bytes(), xlsx.name: xlsx.read_bytes()}
    zip_bytes = build_assets_zip(figs, {slug(k): v for k, v in tables.items()}, files)
    return {**files, f"crisp_dm_{out['run_id']}_completo.zip": zip_bytes}


def render_export_buttons(out: dict[str, Any], key: str) -> None:
    st.subheader("Exportar reporte")
    st.caption("Mismas tablas, gráficos e interpretaciones que en esta página, organizados por fase CRISP-DM. "
               "El paquete completo incluye cada gráfico en PNG de alta resolución (4800×2700), SVG vectorial y HTML interactivo, "
               "cada tabla en CSV y Excel, y el PDF y Excel del reporte.")
    tmp = Path(tempfile.gettempdir()) / "streamlit_app_reports"
    tmp.mkdir(exist_ok=True)
    state_key = f"report_files_{key}_{out['run_id']}"
    if st.button("Generar reporte (PDF + Excel + paquete ZIP)", key=f"gen_{key}"):
        with st.spinner("Renderizando gráficos en alta resolución…"):
            st.session_state[state_key] = _build(out, tmp)
    files = st.session_state.get(state_key)
    if files:
        mimes = {"pdf": "application/pdf", "zip": "application/zip",
                 "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
        for col, (name, data) in zip(st.columns(len(files)), files.items()):
            col.download_button(f"⬇ {name.rsplit('.', 1)[1].upper()}", data, file_name=name, mime=mimes[name.rsplit(".", 1)[1]],
                                key=f"dl_{key}_{name}")
