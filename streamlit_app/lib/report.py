"""Report export buttons, reusing backend/app/reports/ml_analysis_report.py as-is
(pure functions over the same `output` dict shown on screen — no FastAPI/DB involved)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from lib.paths import bootstrap

bootstrap()

from backend.app.reports.ml_analysis_report import generate_ml_pdf, generate_ml_xlsx  # noqa: E402


def render_export_buttons(out: dict[str, Any], key: str) -> None:
    st.subheader("Exportar reporte")
    st.caption("Mismas tablas, gráficos e interpretaciones que en esta página, organizados por fase CRISP-DM.")
    col1, col2 = st.columns(2)
    tmp = Path(tempfile.gettempdir()) / "streamlit_app_reports"
    tmp.mkdir(exist_ok=True)
    with col1:
        if st.button("Generar PDF", key=f"gen_pdf_{key}"):
            path = tmp / f"crisp_dm_{out['run_id']}.pdf"
            generate_ml_pdf(out, path, engine="streamlit")
            st.session_state[f"pdf_path_{key}"] = str(path)
        pdf_path = st.session_state.get(f"pdf_path_{key}")
        if pdf_path and Path(pdf_path).exists():
            st.download_button("Descargar PDF", Path(pdf_path).read_bytes(), file_name=Path(pdf_path).name,
                               mime="application/pdf", key=f"dl_pdf_{key}")
    with col2:
        if st.button("Generar Excel", key=f"gen_xlsx_{key}"):
            path = tmp / f"crisp_dm_{out['run_id']}.xlsx"
            generate_ml_xlsx(out, path)
            st.session_state[f"xlsx_path_{key}"] = str(path)
        xlsx_path = st.session_state.get(f"xlsx_path_{key}")
        if xlsx_path and Path(xlsx_path).exists():
            st.download_button("Descargar Excel", Path(xlsx_path).read_bytes(), file_name=Path(xlsx_path).name,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_xlsx_{key}")
